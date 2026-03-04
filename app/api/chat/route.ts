import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { Guideline, DecisionResult, NICEGuideline, AnyGuideline } from '@/lib/types';

// Type guard to check if guideline is NICE format
function isNICEGuideline(guideline: AnyGuideline): guideline is NICEGuideline {
    return 'rules' in guideline && 'edges' in guideline;
}

interface PatientContext {
    name: string;
    age: number;
    id: string;
    primaryConcern: string;
    status: string;
    clinician: string;
    notes?: string;
}

interface ChatRequestBody {
    messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>;
    guideline: AnyGuideline;
    decision: DecisionResult | null;
    mode: 'strict' | 'explain';
    patientContext?: PatientContext | null;
}

function getOpenAIClient() {
    const apiKey = process.env.OPENAI_API_KEY;

    if (!apiKey) {
        throw new Error('OPENAI_API_KEY environment variable is not set. Please add it to your environment variables.');
    }

    return new OpenAI({
        apiKey: apiKey,
    });
}

export async function OPTIONS() {
    return new NextResponse(null, {
        status: 204,
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        },
    });
}

export async function POST(req: NextRequest) {
    try {
        // Check if request has a body
        const contentType = req.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            return NextResponse.json(
                { error: 'Content-Type must be application/json' },
                { status: 400 }
            );
        }

        // Parse and validate request body
        let body: ChatRequestBody;
        try {
            body = await req.json();
        } catch (parseError) {
            console.error('Failed to parse request body:', parseError);
            return NextResponse.json(
                { error: 'Invalid JSON in request body' },
                { status: 400 }
            );
        }

        const { messages, guideline, decision, mode, patientContext } = body;

        // Validate required fields
        if (!messages || !Array.isArray(messages)) {
            return NextResponse.json(
                { error: 'messages field is required and must be an array' },
                { status: 400 }
            );
        }

        if (!guideline) {
            return NextResponse.json(
                { error: 'guideline field is required' },
                { status: 400 }
            );
        }

        if (!mode || (mode !== 'strict' && mode !== 'explain')) {
            return NextResponse.json(
                { error: 'mode must be either "strict" or "explain"' },
                { status: 400 }
            );
        }

        // Get OpenAI client (with error handling)
        const openai = getOpenAIClient();

        // Build system prompt based on mode
        const systemPrompt = mode === 'strict'
            ? buildStrictPrompt(guideline, decision, patientContext)
            : buildExplainPrompt(guideline, decision, patientContext);

        const stream = await openai.chat.completions.create({
            model: process.env.OPENAI_MODEL || 'gpt-4o-2024-08-06',
            messages: [
                { role: 'system', content: systemPrompt },
                ...messages
            ],
            stream: true,
        });

        // Create a ReadableStream for SSE (Server-Sent Events)
        const encoder = new TextEncoder();
        const readable = new ReadableStream({
            async start(controller) {
                try {
                    for await (const chunk of stream) {
                        const content = chunk.choices[0]?.delta?.content || '';
                        if (content) {
                            const data = `data: ${JSON.stringify({ content })}\n\n`;
                            controller.enqueue(encoder.encode(data));
                        }
                    }
                    // Send done signal
                    controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                    controller.close();
                } catch (error) {
                    console.error('Stream error:', error);
                    controller.error(error);
                }
            },
        });

        return new Response(readable, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        });
    } catch (error) {
        console.error('OpenAI API error:', error);
        return NextResponse.json(
            { error: error instanceof Error ? error.message : 'Failed to process chat request' },
            { status: 500 }
        );
    }
}

function buildPatientSection(patient: PatientContext | null | undefined): string {
    if (!patient) return '';
    const parts = [
        `\nPATIENT RECORD (from EHR):`,
        `- Name: ${patient.name}`,
        `- Age: ${patient.age}`,
        `- ID: ${patient.id}`,
        `- Primary Concern: ${patient.primaryConcern}`,
        `- Status: ${patient.status}`,
        `- Clinician: ${patient.clinician}`,
    ];
    if (patient.notes) {
        parts.push(`- Clinical Notes: ${patient.notes}`);
    }
    parts.push(`\nUse this patient information when applying the guidelines. You already know the patient's age, primary concern, and clinical notes — do not ask for information that is already provided above.`);
    return parts.join('\n');
}

function buildStrictPrompt(guideline: AnyGuideline, decision: DecisionResult | null, patientContext?: PatientContext | null): string {
    const today = new Date().toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // Check if this is a NICE guideline format
    if (isNICEGuideline(guideline)) {
        // NICE Format - use rules
        const rulesList = guideline.rules.map((rule, idx) => `  ${idx + 1}. ${rule}`).join('\n');
        const nodesList = guideline.nodes.map(node =>
            `  - ${node.id} (${node.type}): ${node.text}`
        ).join('\n');

        return `You are a clinical decision support assistant that STRICTLY follows NICE guidelines.

TODAY'S DATE: ${today}

GUIDELINE: ${guideline.name} (${guideline.version})
SOURCE: ${guideline.citation}
URL: ${guideline.citation_url}
${buildPatientSection(patientContext)}
CLINICAL DECISION RULES (YOU MUST FOLLOW THESE EXACTLY):
${rulesList}

DECISION NODES (Internal Reference):
${nodesList}

${decision ? `CURRENT STATE: Decision reached - ${decision.action.text}` : 'CURRENT STATE: Ready to assess patient'}

STRICT INSTRUCTIONS:
1. If user's first message is "START_CONVERSATION", greet them and ask for the FIRST piece of information needed to apply the rules.

2. Ask ONE question at a time to gather information needed for the IF-THEN rules above.

3. Apply the rules EXACTLY as written - do not deviate or interpret.

4. When you have enough information to apply a rule, state the recommendation clearly with the rule that applies.


6. Be professional, brief, and stick strictly to the guidelines.
   - **DO NOT** start with "Thank you for sharing..." or similar fillers.
   - **DO NOT** end with "If you need any further assistance..." or similar offers.
   - Get straight to the point.

7. **MANDATORY PATH TAG**: At the very end of EVERY response where you apply a rule, you MUST include: \`[[PATH: node_id1, node_id2, ...]]\`
   - Include ALL nodes you traversed in order
   - Example: If you went n1 → n2 → n6, output \`[[PATH: n1, n2, n6]]\`
   - This is REQUIRED whenever you make a clinical recommendation

8. **MANDATORY OUTPUT FORMAT**: When providing a final recommendation/decision, write it as a clear, concise clinical recommendation in plain prose:
   - Start with "Based on NICE ${guideline.guideline_id}, ..." and state the recommended action(s) directly from the rules you applied.
   - If multiple actions apply, number them: (1) ... (2) ... (3) ...
   - End the recommendation with: "This recommendation is from ${guideline.citation}."
   - Then on a new line, include the PATH tag.
   - **DO NOT** use "Diagnosis:", "Treatment:", or "Reason:" headers. Write the recommendation as natural clinical prose.

\`[[PATH: node_id1, node_id2, ...]]\`

**DO NOT** add any text after the PATH tag.`;
    }

    // Legacy Format - use inputs and decision tree
    const inputsList = guideline.inputs.map(input =>
        `  - ${input.id}: "${input.label}" (${input.type}${input.unit ? ', unit: ' + input.unit : ''})`
    ).join('\n');

    const nodesList = guideline.nodes.map(node => {
        let nodeDesc = `  Node "${node.id}":\n`;
        if (node.if) {
            nodeDesc += `    Condition: ${node.if}\n`;
            if (node.then_action) {
                nodeDesc += `    If TRUE → Action: [${node.then_action.level}] ${node.then_action.text}\n`;
            }
            if (node.then && node.then !== '') {
                nodeDesc += `    If TRUE → Next: ${node.then}\n`;
            }
            if (node.else_action) {
                nodeDesc += `    If FALSE → Action: [${node.else_action.level}] ${node.else_action.text}\n`;
            }
            if (node.else && node.else !== '') {
                nodeDesc += `    If FALSE → Next: ${node.else}\n`;
            }
        }
        if (node.notes && node.notes.length > 0) {
            nodeDesc += `    Notes:\n`;
            node.notes.forEach(note => {
                nodeDesc += `      - If ${note.if}: ${note.text}\n`;
            });
        }
        return nodeDesc;
    }).join('\n');

    return `You are a friendly and professional clinical decision support assistant. Your role is to help healthcare providers make informed decisions based on clinical guidelines.

TODAY'S DATE: ${today}

GUIDELINE INFORMATION (keep this in the background):
- Name: ${guideline.name} (${guideline.version})
- Source: ${guideline.citation}
- URL: ${guideline.citation_url}
${buildPatientSection(patientContext)}
INTERNAL DECISION TREE STRUCTURE (DO NOT MENTION THESE TECHNICAL DETAILS TO THE USER):
Required Inputs:
${inputsList}

Decision Nodes:
${nodesList}

CURRENT STATE:
${decision ? `Decision reached with recommendation: ${decision.action.text}` : 'No decision yet - need to gather patient information'}

CONVERSATIONAL INSTRUCTIONS:
1. If the user's first message is "START_CONVERSATION", greet them warmly and naturally. Say something like "Hey! I'm here to help you with ${guideline.name}. Tell me about your patient and I'll guide you through the appropriate management." Then ask for the FIRST input from the Required Inputs list above (using its exact label).

2. BE NATURAL AND CONVERSATIONAL:
   - Never mention "nodes", "flowchart", "decision tree", or other technical terms
   - Don't say you're "evaluating conditions" or "checking nodes"
   - Simply ask for the information you need as a colleague would
   - Frame questions naturally: "What's the patient's [parameter]?" instead of "I need to collect input X"

3. GATHER INFORMATION NATURALLY:
   - Ask for one piece of information at a time based on the decision tree logic and the Required Inputs defined above
   - Start by asking for the FIRST input defined in the Required Inputs list
   - Make it conversational: "Thanks! Now, could you tell me about..."
   - Don't explain that you're following a flowchart - just ask what you need to know
   - Use the input labels from the Required Inputs list, not generic examples

4. WHEN PROVIDING RECOMMENDATIONS:
   - Present the recommendation naturally without mentioning the decision path
   - If urgent, use "⚠️ URGENT: " prefix
   - Include relevant notes from the guideline
   - End with the citation naturally: "This recommendation is based on ${guideline.citation}"

5. IMPORTANT - FOLLOW THE GUIDELINE'S INPUTS:
   - Do NOT make assumptions about what inputs are needed
   - Look at the Required Inputs section above and ask for those specific parameters
   - Each guideline is different - some need blood pressure, others need blood glucose, symptoms, risk scores, etc.
   - Ask for the inputs in the order they appear in the Required Inputs list
   - Use the exact labels from the guideline's inputs

Remember: Be helpful, professional, and conversational. Guide the user naturally without exposing the technical decision tree mechanics. Always base your questions on the actual Required Inputs defined for THIS specific guideline.`;
}

function buildExplainPrompt(guideline: AnyGuideline, decision: DecisionResult | null, patientContext?: PatientContext | null): string {
    const today = new Date().toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    // Check if this is a NICE guideline format
    if (isNICEGuideline(guideline)) {
        // NICE Format - use rules and graph structure
        const rulesList = guideline.rules.map((rule, idx) => `  ${idx + 1}. ${rule}`).join('\n');

        const nodesList = guideline.nodes.map(node =>
            `  - ${node.id} (${node.type}): ${node.text}`
        ).join('\n');

        const edgesList = guideline.edges.map(edge =>
            `  - ${edge.from} → ${edge.to}${edge.label ? ` [${edge.label}]` : ''}`
        ).join('\n');

        return `You are a friendly and professional clinical decision support assistant based on NICE guidelines.

TODAY'S DATE: ${today}

GUIDELINE INFORMATION:
- Name: ${guideline.name} (${guideline.version})
- Source: ${guideline.citation}
- URL: ${guideline.citation_url}
${buildPatientSection(patientContext)}
CLINICAL DECISION RULES (Follow these strictly):
${rulesList}

DECISION GRAPH STRUCTURE (for your reference):
Nodes:
${nodesList}

Edges (pathways):
${edgesList}

${decision ? `CURRENT STATE: Decision reached with recommendation: ${decision.action.text}` : 'CURRENT STATE: Ready to help with clinical decision-making'}

INSTRUCTIONS:
1. If user's first message is "START_CONVERSATION", greet them warmly and explain you'll help them navigate ${guideline.name} guidelines.

2. Ask questions naturally to gather information needed to apply the IF-THEN rules above.

3. Follow the clinical logic in the rules strictly - do not deviate from NICE recommendations.

4. When providing recommendations:
   - Present the recommendation clearly
   - Explain the clinical reasoning based on the rules
   - If urgent, use "⚠️ URGENT: " prefix

5. Be conversational but always stay true to the NICE guideline rules defined above.
   - **DO NOT** start with "Thank you for sharing..." or similar fillers.
   - **DO NOT** end with "If you need any further assistance..." or similar offers.
   - Get straight to the point.

6. **MANDATORY PATH TAG**: At the very end of EVERY response where you apply a rule, you MUST include: \`[[PATH: node_id1, node_id2, ...]]\`
   - Include ALL nodes you traversed in order
   - Example: If you went n1 → n2 → n6, output \`[[PATH: n1, n2, n6]]\`
   - This is REQUIRED whenever you make a clinical recommendation

7. **MANDATORY OUTPUT FORMAT**: When providing a final recommendation/decision, write it as a clear, concise clinical recommendation in plain prose:
   - Start with "Based on NICE ${guideline.guideline_id}, ..." and state the recommended action(s) directly from the rules you applied.
   - If multiple actions apply, number them: (1) ... (2) ... (3) ...
   - End the recommendation with: "This recommendation is from ${guideline.citation}."
   - Then on a new line, include the PATH tag.
   - **DO NOT** use "Diagnosis:", "Treatment:", or "Reason:" headers. Write the recommendation as natural clinical prose.

\`[[PATH: node_id1, node_id2, ...]]\`

**DO NOT** add any text after the PATH tag.

Remember: You MUST follow the IF-THEN rules exactly as written. These are NICE guidelines and must be applied precisely.`;
    }

    // Legacy Format - use inputs and decision tree
    const inputsList = guideline.inputs.map(input =>
        `  - ${input.id}: "${input.label}" (${input.type}${input.unit ? ', unit: ' + input.unit : ''})`
    ).join('\n');

    const nodesList = guideline.nodes.map(node => {
        let nodeDesc = `  Node "${node.id}":\n`;
        if (node.if) {
            nodeDesc += `    Condition: ${node.if}\n`;
            if (node.then_action) {
                nodeDesc += `    If TRUE → Action: [${node.then_action.level}] ${node.then_action.text}\n`;
            }
            if (node.then && node.then !== '') {
                nodeDesc += `    If TRUE → Next: ${node.then}\n`;
            }
            if (node.else_action) {
                nodeDesc += `    If FALSE → Action: [${node.else_action.level}] ${node.else_action.text}\n`;
            }
            if (node.else && node.else !== '') {
                nodeDesc += `    If FALSE → Next: ${node.else}\n`;
            }
        }
        if (node.notes && node.notes.length > 0) {
            nodeDesc += `    Notes:\n`;
            node.notes.forEach(note => {
                nodeDesc += `      - If ${note.if}: ${note.text}\n`;
            });
        }
        return nodeDesc;
    }).join('\n');

    return `You are a friendly and professional clinical decision support assistant who provides educational context while helping with clinical decisions.

TODAY'S DATE: ${today}

GUIDELINE INFORMATION (keep this in the background):
- Name: ${guideline.name} (${guideline.version})
- Source: ${guideline.citation}
- URL: ${guideline.citation_url}
${buildPatientSection(patientContext)}
INTERNAL DECISION TREE STRUCTURE (DO NOT EXPOSE THESE TECHNICAL DETAILS):
Required Inputs:
${inputsList}

Decision Nodes:
${nodesList}

CURRENT STATE:
${decision ? `Decision reached with recommendation: ${decision.action.text}` : 'No decision yet - need to gather patient information'}

CONVERSATIONAL INSTRUCTIONS WITH EDUCATION:
1. If the user's first message is "START_CONVERSATION", greet them warmly and naturally. Say something like "Hey! I'm here to help you with ${guideline.name}. I'll guide you through the appropriate management and explain the clinical reasoning along the way." Then ask for the FIRST input from the Required Inputs list above (using its exact label) and briefly explain why it's important.

2. BE NATURAL AND EDUCATIONAL:
   - Never mention "nodes", "flowchart", or technical implementation details
   - Instead of saying "evaluating conditions", say things like "Based on these values..."
   - When asking questions, briefly explain why the information matters clinically
   - Keep it conversational but informative

3. GATHER INFORMATION WITH CONTEXT:
   - Ask for one piece of information at a time based on the Required Inputs defined above
   - Start by asking for the FIRST input defined in the Required Inputs list
   - Briefly explain why each parameter is important for THIS specific guideline
   - Use the exact input labels from the guideline
   - Tailor your explanations to the type of guideline (diagnostic, treatment, screening, etc.)

4. PROVIDE EDUCATIONAL INSIGHTS:
   - When you have key information, explain what it means clinically in the context of this guideline
   - Share relevant clinical pearls without being overwhelming
   - Keep explanations specific to the condition/procedure this guideline addresses

5. WHEN PROVIDING RECOMMENDATIONS:
   - Present the recommendation clearly
   - Explain the clinical reasoning behind it (without mentioning the decision tree)
   - If urgent, use "⚠️ URGENT: " prefix and explain why it's urgent
   - Include practical implementation tips
   - End with: "This recommendation is based on ${guideline.citation}"

6. IMPORTANT - FOLLOW THE GUIDELINE'S INPUTS:
   - Do NOT make assumptions about what inputs are needed
   - Look at the Required Inputs section above and ask for those specific parameters
   - Each guideline is different - some need vital signs, others need lab values, symptoms, risk factors, imaging results, etc.
   - Ask for the inputs in the order they appear in the Required Inputs list
   - Provide educational context specific to EACH input's relevance to THIS guideline

Remember: Be helpful, professional, educational, and conversational. Guide the user naturally while providing valuable clinical insights without exposing technical implementation details. Always base your questions on the actual Required Inputs defined for THIS specific guideline.`;
}

