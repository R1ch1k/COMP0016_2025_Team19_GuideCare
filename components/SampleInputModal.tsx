// @ts-ignore - Custom dialog component
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/shadcn-io/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SampleInputModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SampleInputModal({ isOpen, onClose }: SampleInputModalProps) {
    const samples = [
        {
            title: "NG84 \u2014 Sore Throat",
            patient: "Samantha Chen (32F, asthma, allergic to Sulfa)",
            scenario: "Samantha has a sore throat with fever 38.5\u00B0C, purulent tonsils, and tender lymph nodes. FeverPAIN score is 4.",
            guideline: "NG84",
            clarification: "Does the patient have any signs of severe systemic infection?",
            sampleAnswer: "No, she does not have any signs of severe systemic infection or complications.",
            expected: "Consider an immediate antibiotic or a back-up antibiotic prescription. Advise paracetamol/ibuprofen, adequate fluids, and medicated lozenges.",
            color: "bg-red-50 border-red-200",
        },
        {
            title: "NG136 \u2014 Hypertension",
            patient: "Alex Morgan (46M, Type 2 Diabetes, on Metformin)",
            scenario: "Alex presents with a clinic BP reading of 160/100 mmHg at a routine check-up. He has no symptoms.",
            guideline: "NG136",
            clarification: "Has ambulatory blood pressure monitoring (ABPM) been performed?",
            sampleAnswer: "Yes, ABPM was done. Daytime average is 152/95 mmHg.",
            expected: "Offer antihypertensive drug treatment. Consider ACE inhibitor or ARB as first-line for patients with diabetes.",
            color: "bg-orange-50 border-orange-200",
        },
        {
            title: "NG232 \u2014 Head Injury",
            patient: "Jordan Lee (28M, no significant history)",
            scenario: "Jordan fell off his bike and hit his head on the pavement. He vomited twice and has a persistent headache. No loss of consciousness. GCS is 15.",
            guideline: "NG232",
            clarification: "Are there any signs of basal skull fracture or suspected cervical spine injury?",
            sampleAnswer: "No signs of basal skull fracture or cervical spine injury.",
            expected: "Perform CT head scan within 1 hour due to persistent vomiting (>1 episode). Monitor for neurological deterioration.",
            color: "bg-blue-50 border-blue-200",
        },
        {
            title: "NG91 \u2014 Otitis Media",
            patient: "Any patient (child, 5 years old)",
            scenario: "5-year-old child presenting with acute ear pain for 2 days, fever of 38.2\u00B0C, and mild redness of the tympanic membrane. No ear discharge.",
            guideline: "NG91",
            clarification: "Is the child systemically very unwell or showing signs of a more serious condition?",
            sampleAnswer: "No, the child is not systemically unwell. No signs of mastoiditis or meningitis.",
            expected: "Consider a no-antibiotic or back-up antibiotic prescribing strategy. Advise paracetamol or ibuprofen for pain. Reassess if symptoms worsen.",
            color: "bg-green-50 border-green-200",
        },
        {
            title: "NG112 \u2014 Recurrent UTI",
            patient: "Any patient (55F, postmenopausal)",
            scenario: "55-year-old postmenopausal woman presenting with her third UTI in 6 months. Symptoms include dysuria and urinary frequency. No fever.",
            guideline: "NG112",
            clarification: "Does the patient have any signs of pyelonephritis such as fever, flank pain, or chills?",
            sampleAnswer: "No, she has no fever, flank pain, or systemic symptoms.",
            expected: "Consider antibiotic prophylaxis for recurrent UTI. Discuss vaginal oestrogen if appropriate. Review behavioural and personal hygiene measures.",
            color: "bg-purple-50 border-purple-200",
        },
    ];

    return (
        <Dialog open={isOpen as boolean} onOpenChange={(open) => !open && onClose()}>
            <DialogContent className="max-w-3xl">
                <DialogHeader>
                    <DialogTitle>Sample Inputs & Expected Outcomes</DialogTitle>
                    <p className="sr-only">
                        Step-by-step examples showing how to use the clinical decision support system with different NICE guidelines.
                    </p>
                </DialogHeader>
                <ScrollArea className="max-h-[70vh] pr-4">
                    <div className="space-y-4">
                        <p className="text-sm text-gray-600">
                            Follow these examples to test the pipeline. Select a patient, describe symptoms, answer any clarification questions, and receive a guideline-based recommendation.
                        </p>
                        <div className="grid gap-4">
                            {samples.map((sample, index) => (
                                <div
                                    key={index}
                                    className={`p-4 rounded-lg border ${sample.color} transition-all hover:shadow-sm`}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <h3 className="font-semibold text-gray-900">{sample.title}</h3>
                                        <span className="text-xs font-mono bg-white/70 px-2 py-0.5 rounded border border-gray-200">
                                            {sample.guideline}
                                        </span>
                                    </div>
                                    <div className="space-y-2">
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Patient</span>
                                            <p className="text-sm text-gray-700 mt-0.5">{sample.patient}</p>
                                        </div>
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Step 1 &mdash; Describe Symptoms</span>
                                            <p className="text-sm text-gray-800 mt-0.5 bg-white/50 rounded px-2 py-1 font-mono text-xs leading-relaxed">{sample.scenario}</p>
                                        </div>
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Step 2 &mdash; System Asks</span>
                                            <p className="text-sm text-gray-700 mt-0.5 italic">{sample.clarification}</p>
                                        </div>
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Step 3 &mdash; Your Reply</span>
                                            <p className="text-sm text-gray-800 mt-0.5 bg-white/50 rounded px-2 py-1 font-mono text-xs leading-relaxed">{sample.sampleAnswer}</p>
                                        </div>
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Expected Recommendation</span>
                                            <p className="text-sm font-medium text-gray-900 mt-0.5">{sample.expected}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
