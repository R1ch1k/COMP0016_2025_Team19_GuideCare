import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const SYSTEM_PROMPT = `You are an expert clinical assistant for NHS doctors and healthcare professionals. You help with:
- General medical questions and clinical reasoning
- Explaining NICE guidelines, protocols, and evidence
- Drug interactions, dosing, and prescribing queries
- Interpreting clinical findings and investigations
- Differential diagnoses and management plans
- Medical education and concept explanations

Guidelines:
- Be concise, accurate, and clinically focused
- Cite NICE guidelines, BNF, or evidence-based sources when relevant
- Flag urgent/safety-critical information clearly
- Do not diagnose specific patients — you are a general reference assistant
- If a question is outside your knowledge or requires direct patient assessment, say so clearly`;

function getOpenAIClient() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
        throw new Error('OPENAI_API_KEY environment variable is not set.');
    }
    return new OpenAI({ apiKey });
}

export async function POST(req: NextRequest) {
    try {
        const contentType = req.headers.get('content-type');
        if (!contentType?.includes('application/json')) {
            return NextResponse.json({ error: 'Content-Type must be application/json' }, { status: 400 });
        }

        const body = await req.json();
        const { messages } = body;

        if (!messages || !Array.isArray(messages)) {
            return NextResponse.json({ error: 'messages field is required' }, { status: 400 });
        }

        // Validate messages: only allow user/assistant roles with string content
        const sanitizedMessages = (messages as { role: string; content: unknown }[])
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .filter((m) => typeof m.content === 'string' && m.content.trim().length > 0)
            .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content as string }));

        if (sanitizedMessages.length === 0) {
            return NextResponse.json({ error: 'No valid messages provided' }, { status: 400 });
        }

        const openai = getOpenAIClient();

        const stream = await openai.chat.completions.create({
            model: process.env.OPENAI_MODEL || 'gpt-4o',
            messages: [
                { role: 'system', content: SYSTEM_PROMPT },
                ...sanitizedMessages,
            ],
            stream: true,
        });

        const encoder = new TextEncoder();
        const readable = new ReadableStream({
            async start(controller) {
                try {
                    for await (const chunk of stream) {
                        const content = chunk.choices[0]?.delta?.content || '';
                        if (content) {
                            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content })}\n\n`));
                        }
                    }
                    controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                    controller.close();
                } catch (error) {
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
        console.error('Error handling /api/general-chat request:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
