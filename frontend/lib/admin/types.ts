export interface WorkflowInfo {
  name: string;
  functional: boolean;
}

export interface AdminSettings {
  default_workflow: string;
  workflows: WorkflowInfo[];
}
