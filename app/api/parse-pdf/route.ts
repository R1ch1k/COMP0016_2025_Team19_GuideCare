import { NextRequest, NextResponse } from 'next/server';
import OpenAI, { toFile } from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';
import { PDF_UPLOAD_CONFIG } from '@/lib/config/pdf-upload';

function getOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY environment variable is not set. Please add it to your environment variables.');
  }
  
  return new OpenAI({
    apiKey: apiKey,
  });
}

// Define the Zod schema for NICE guideline structure with IF-THEN rules and decision graph
// Note: All fields must be required for OpenAI structured outputs
const GuidelineSchema = z.object({
  guideline_id: z.string(),
  name: z.string(),
  version: z.string(),
  citation: z.string(),
  citation_url: z.string(),
  rules: z.array(z.string()), // Array of IF-THEN rule strings
  nodes: z.array(
    z.object({
      id: z.string(),
      type: z.enum(['condition', 'action']),
      text: z.string(),
    })
  ),
  edges: z.array(
    z.object({
      from: z.string(),
      to: z.string(),
      label: z.string(), // Required - use empty string if no label needed
    })
  ),
});

// Define the Zod schema for condition evaluators (recursive for nested conditions)
const ConditionEvaluatorSchema: z.ZodType<any> = z.lazy(() => z.union([
  z.object({ variable: z.string() }), // Boolean
  z.object({
    type: z.literal("bp_compare"),
    variable: z.string(),
    threshold: z.string(),
    op: z.enum([">=", ">", "<=", "<", "=="])
  }),
  z.object({
    type: z.literal("bp_range"),
    variable: z.string(),
    systolic_min: z.number(),
    systolic_max: z.number(),
    diastolic_min: z.number(),
    diastolic_max: z.number()
  }),
  z.object({
    type: z.literal("age_compare"),
    variable: z.string(),
    threshold: z.number(),
    op: z.enum([">=", ">", "<=", "<", "=="])
  }),
  z.object({
    type: z.literal("numeric_compare"),
    variable: z.string(),
    threshold: z.number(),
    op: z.enum([">=", ">", "<=", "<", "=="])
  }),
  z.object({
    type: z.literal("or"),
    conditions: z.array(ConditionEvaluatorSchema)
  }),
  z.object({
    type: z.literal("and"),
    conditions: z.array(ConditionEvaluatorSchema)
  }),
]));


export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Get OpenAI client (with error handling)
    const openai = getOpenAIClient();

    // Convert File to a format OpenAI accepts
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    // Upload the PDF file to OpenAI
    const uploadedFile = await openai.files.create({
      file: await toFile(buffer, file.name, { type: file.type }),
      purpose: PDF_UPLOAD_CONFIG.openai.filePurpose,
    });

    // Create an assistant with file search capability
    const assistant = await openai.beta.assistants.create({
      name: PDF_UPLOAD_CONFIG.openai.assistantName,
      instructions: PDF_UPLOAD_CONFIG.systemPrompt,
      model: PDF_UPLOAD_CONFIG.openai.model,
      tools: [{ type: 'file_search' }],
      tool_resources: {
        file_search: {
          vector_stores: [{
            file_ids: [uploadedFile.id]
          }]
        }
      },
      response_format: zodResponseFormat(GuidelineSchema, 'medical-guideline'),
    });

    // Create a thread and run
    const thread = await openai.beta.threads.create({
      messages: [{
        role: 'user',
        content: 'Please analyze this medical guideline PDF and extract the structured information.'
      }]
    });

    const run = await openai.beta.threads.runs.createAndPoll(thread.id, {
      assistant_id: assistant.id,
    });

    if (run.status !== 'completed') {
      throw new Error(`Run failed with status: ${run.status}`);
    }

    // Get the messages
    const messages = await openai.beta.threads.messages.list(thread.id);
    const assistantMessage = messages.data.find(m => m.role === 'assistant');
    
    // Clean up
    try {
      await openai.beta.assistants.delete(assistant.id);
      await openai.files.delete(uploadedFile.id);
    } catch (deleteError) {
      console.warn('Failed to clean up resources:', deleteError);
    }

    if (!assistantMessage || !assistantMessage.content[0] || assistantMessage.content[0].type !== 'text') {
      return NextResponse.json(
        { error: 'No response from assistant' },
        { status: 500 }
      );
    }

    const guideline = JSON.parse(assistantMessage.content[0].text.value);

    if (!guideline) {
      return NextResponse.json(
        { error: 'No guideline data extracted from PDF' },
        { status: 500 }
      );
    }

    // STEP 2: Generate condition evaluators

    try {
      const evaluatorCompletion = await openai.chat.completions.create({
        model: PDF_UPLOAD_CONFIG.openai.model,
        messages: [
          {
            role: 'system',
            content: PDF_UPLOAD_CONFIG.evaluatorPrompt,
          },
          {
            role: 'user',
            content: `Analyze this guideline JSON and generate condition_evaluators:\n\n${JSON.stringify({ nodes: guideline.nodes }, null, 2)}`,
          },
        ],
        response_format: { type: "json_object" }, // Use regular JSON mode instead of structured outputs
      });

      const evaluatorsResponse = evaluatorCompletion.choices[0].message.content;

      if (evaluatorsResponse) {
        const evaluatorsData = JSON.parse(evaluatorsResponse);

        // Validate the structure manually
        if (evaluatorsData.condition_evaluators && typeof evaluatorsData.condition_evaluators === 'object') {
          // Merge evaluators into the guideline
          guideline.condition_evaluators = evaluatorsData.condition_evaluators;

          // Evaluators generated successfully
        } else {
          console.warn('Evaluators response missing condition_evaluators field');
        }
      }
    } catch (evaluatorError) {
      console.warn('Failed to generate evaluators, continuing without them:', evaluatorError);
      // Continue without evaluators rather than failing the entire request
    }

    return NextResponse.json({ guideline });
  } catch (error) {
    console.error('PDF parsing error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to parse PDF' },
      { status: 500 }
    );
  }
}

// Mark as Node.js runtime (not Edge)
export const runtime = 'nodejs';

