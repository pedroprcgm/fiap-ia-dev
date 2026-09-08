export interface Patient {
  patient_id: string;
  name: string;
  main_condition: string;
}

export interface AskResponse {
  response_text: string;
  sources: string[];
  pending_exam_notice: string | null;
}
