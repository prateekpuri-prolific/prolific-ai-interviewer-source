from app.schemas.agent import InterviewGuide
from app.schemas.study import ResearchSpec
from studies.frontend_ai_failures_v03.protocol import concrete_guide


def test_concrete_guide_is_valid_and_aligns_research_targets_without_mutating_input():
    original={'study':{'research_spec':ResearchSpec().model_dump(mode='json')},
              'agent':{'interview_guide':{'orchestration_profile':'exploratory',
                                         'questions':[{'id':'old','text':'Old question?'}]}}}
    result=concrete_guide(original)
    guide=InterviewGuide.model_validate(result['agent']['interview_guide'])
    spec=ResearchSpec.model_validate(result['study']['research_spec'])
    assert original['agent']['interview_guide']['questions'][0]['id']=='old'
    assert {q.id for q in guide.questions}=={t.id for t in spec.targets}
    assert guide.conversation_policy=='reflective_v1'
    assert 'input_inventory' in {q.id for q in guide.questions}
    assert 'reference_context' in {q.id for q in guide.questions}
    assert guide.questions[0].text in result['agent']['welcome_message']
    assert 'ready to begin' not in result['agent']['welcome_message'].lower()
    assert result['agent']['widget_show_progress'] is False
