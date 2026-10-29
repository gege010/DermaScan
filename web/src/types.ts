export type Assessment = {
  id: string;
  assessment: {
    status: 'urgent' | 'review' | 'unsupported' | 'retake' | 'research';
    title: string;
    summary: string;
    next_steps: string[];
  };
  context: { symptoms: string[]; duration: string; area: string; image_kind: string };
  quality: { width: number; height: number; issues: string[]; scope: string } | null;
  candidates: { code: string; name: string; score: number }[];
  education: { title: string; body: string }[];
  references: { title: string; url: string }[];
  warnings: string[];
  privacy: string;
  model: {
    available: boolean;
    mode: string;
    clinical_validation: boolean;
    class_count: number;
    message: string;
  };
  elapsed_ms: number;
};
export type JournalEntry = {
  id: string;
  date: string;
  area: string;
  feeling: string;
  note: string;
};
