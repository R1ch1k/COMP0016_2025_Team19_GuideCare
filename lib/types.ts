// NICE Guideline Types (New Format - IF-THEN Rules + Graph)
export interface NICEGraphNode {
  id: string;
  type: "condition" | "action";
  text: string;
}

export interface NICEGraphEdge {
  from: string;
  to: string;
  label?: string;
}

// Condition Evaluator Types
export type ConditionEvaluator =
  | { variable: string } // Boolean check
  | { type: "bp_compare"; variable: string; threshold: string; op: ">=" | ">" | "<=" | "<" | "==" }
  | { type: "bp_range"; variable: string; systolic_min: number; systolic_max: number; diastolic_min: number; diastolic_max: number }
  | { type: "age_compare"; variable: string; threshold: number; op: ">=" | ">" | "<=" | "<" | "==" }
  | { type: "numeric_compare"; variable: string; threshold: number; op: ">=" | ">" | "<=" | "<" | "==" }
  | { type: "or"; conditions: ConditionEvaluator[] }
  | { type: "and"; conditions: ConditionEvaluator[] };

export interface ConditionEvaluators {
  [nodeId: string]: ConditionEvaluator;
}

export interface NICEGuideline {
  guideline_id: string;
  name: string;
  version: string;
  citation: string;
  citation_url: string;
  rules: string[]; // Array of IF-THEN rule strings
  nodes: NICEGraphNode[];
  edges: NICEGraphEdge[];
  condition_evaluators?: ConditionEvaluators; // Optional for backwards compatibility
}

// Legacy Types (for backwards compatibility with existing guidelines)
export interface GuidelineInput {
  id: string;
  label: string;
  type: "number" | "boolean" | "text";
  unit?: string;
}

export interface ActionOutput {
  level: "info" | "advice" | "start" | "urgent";
  text: string;
}

export interface ConditionalNote {
  if: string;
  text: string;
}

export interface DecisionNode {
  id: string;
  if?: string;
  then?: string;
  else?: string;
  then_action?: ActionOutput;
  else_action?: ActionOutput;
  notes?: ConditionalNote[];
}

export interface Guideline {
  guideline_id: string;
  name: string;
  version: string;
  citation: string;
  citation_url: string;
  inputs: GuidelineInput[];
  nodes: DecisionNode[];
}

// Union type to support both formats
export type AnyGuideline = NICEGuideline | Guideline;

export interface DecisionResult {
  action: ActionOutput;
  path: string[];
  notes: string[];
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  decision?: DecisionResult;
  pathwayWalked?: string[];
  selectedGuideline?: string;
  isUrgent?: boolean;
}

