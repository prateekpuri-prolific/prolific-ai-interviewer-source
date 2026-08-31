import { describe, expect, it } from "vitest";
import type { InterviewGuideConfig } from "../lib/api";
import { guideForStudyComposer } from "./StudyWorkspacePage";

const question = (text: string) => ({
  id: "question_1",
  module: "Core interview",
  text,
  probes: ["  What happened next?  ", ""],
  max_follow_ups: 2,
  transition: null,
  purpose: "",
  target_ids: [" experience ", ""],
  evidence_goal: [" A concrete example ", ""],
  do_not_assume: [" That it was helpful ", ""],
  priority: "required" as const,
  sensitivity: "standard" as const,
  phase: "core" as const,
  visit_scope: "all" as const,
  role_variants: {},
  returning_role_variants: {},
});

const guide = (texts: string[]): InterviewGuideConfig => ({
  questions: texts.map(question),
  closing_message: "Thank you for taking part.",
  allow_follow_ups: true,
  probe_selection: "relevance",
  allow_emergent_follow_ups: true,
});

describe("guideForStudyComposer", () => {
  it("sends no current guide when a new study only has a blank placeholder", () => {
    expect(guideForStudyComposer(guide(["   "]))).toBeNull();
  });

  it("removes blank cards while preserving and cleaning substantive questions", () => {
    const result = guideForStudyComposer(guide(["A real question?", ""]));

    expect(result?.questions).toHaveLength(1);
    expect(result?.questions[0].text).toBe("A real question?");
    expect(result?.questions[0].probes).toEqual(["What happened next?"]);
    expect(result?.questions[0].target_ids).toEqual(["experience"]);
    expect(result?.questions[0].evidence_goal).toEqual(["A concrete example"]);
    expect(result?.questions[0].do_not_assume).toEqual(["That it was helpful"]);
    expect(result?.allow_follow_ups).toBe(true);
  });
});
