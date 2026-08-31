"""
Tests for the Studies CRUD API endpoints.

Uses the test FastAPI app with in-memory SQLite — no real DB needed.
"""

import uuid

import pytest
from httpx import AsyncClient


class TestStudiesCRUD:
    async def test_list_studies_empty(self, client: AsyncClient):
        resp = await client.get("/api/studies")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_create_study(self, client: AsyncClient):
        resp = await client.post(
            "/api/studies",
            json={"title": "Research Study 1", "description": "First study"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Research Study 1"
        assert data["description"] == "First study"
        assert data["status"] == "draft"
        assert "id" in data

    async def test_create_study_minimal(self, client: AsyncClient):
        resp = await client.post(
            "/api/studies",
            json={"title": "Minimal Study"},
        )
        assert resp.status_code == 201
        assert resp.json()["title"] == "Minimal Study"

    async def test_create_study_empty_title(self, client: AsyncClient):
        resp = await client.post(
            "/api/studies",
            json={"title": ""},
        )
        assert resp.status_code == 422  # Validation error

    async def test_get_study(self, client: AsyncClient):
        # Create
        create_resp = await client.post(
            "/api/studies",
            json={"title": "Get Test"},
        )
        study_id = create_resp.json()["id"]

        # Get
        resp = await client.get(f"/api/studies/{study_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == study_id
        assert resp.json()["title"] == "Get Test"

    async def test_get_study_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/api/studies/{fake_id}")
        assert resp.status_code == 404

    async def test_update_study(self, client: AsyncClient):
        # Create
        create_resp = await client.post(
            "/api/studies",
            json={"title": "Original Title"},
        )
        study_id = create_resp.json()["id"]

        # Update
        resp = await client.patch(
            f"/api/studies/{study_id}",
            json={"title": "Updated Title"},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Updated Title"

    async def test_research_protocol_round_trip_and_validation(self, client: AsyncClient):
        create_resp = await client.post("/api/studies", json={"title": "Protocol Test"})
        study_id = create_resp.json()["id"]
        spec = {
            "version": "0.2",
            "stage": "pilot",
            "method": "exploration",
            "decision_to_inform": "Choose a parent support concept.",
            "research_questions": ["What do parents observe directly?"],
            "population": {
                "participant_role": "Parents",
                "subject": "Children's AI use",
                "knowledge_limits": ["May not see use at school"],
            },
            "targets": [{"id": "direct_observation", "name": "Observed examples"}],
            "clarification_policy": {
                "participant_definition": "AI means a feature that generates content or carries on a conversation.",
                "in_scope": ["Generative chatbot interactions"],
                "out_of_scope": ["Ordinary social-media browsing"],
                "ambiguous_cases": ["Social platforms with optional AI features"],
                "ambiguity_handling": "Ask which feature was used.",
            },
            "participant_information": {
                "purpose_summary": "Understand family experiences with AI.",
                "consent_required": True,
            },
            "analysis_notes": "Private hypothesis that must not reach the agent.",
        }

        resp = await client.patch(
            f"/api/studies/{study_id}", json={"research_spec": spec}
        )
        assert resp.status_code == 200
        saved = resp.json()["research_spec"]
        assert saved["version"] == "0.2"
        assert saved["targets"][0]["id"] == "direct_observation"
        assert saved["clarification_policy"]["in_scope"] == [
            "Generative chatbot interactions"
        ]
        assert saved["participant_information"]["rights"]

        spec["targets"][0]["id"] = "invalid target id"
        invalid = await client.patch(
            f"/api/studies/{study_id}", json={"research_spec": spec}
        )
        assert invalid.status_code == 422

    async def test_update_study_status(self, client: AsyncClient):
        create_resp = await client.post(
            "/api/studies",
            json={
                "title": "Status Test",
                "research_spec": {
                    "participant_information": {
                        "participation_summary": "Answer questions about a recent work task.",
                        "data_handling_summary": (
                            "Audio is not saved; researchers review transcripts."
                        ),
                        "data_controller": "Example Research Ltd",
                        "data_recipients_summary": "The authorized research team.",
                        "risks_summary": "Low risk; questions may be skipped.",
                        "compensation_summary": "Payment is described in the invitation.",
                        "researcher_identity": "Example research team",
                        "researcher_contact": "Contact us through the recruitment platform.",
                        "complaints_contact": "Contact the research ethics team.",
                        "data_retention_period": "Responses are kept for 12 months.",
                        "withdrawal_instructions": "Contact the researcher to request deletion.",
                    },
                    "fielding": {"recruitment_source": "direct_link"},
                },
            },
        )
        study_id = create_resp.json()["id"]

        resp = await client.patch(
            f"/api/studies/{study_id}",
            json={"status": "active"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    async def test_study_cannot_activate_without_data_handling_summary(
        self, client: AsyncClient
    ):
        create_resp = await client.post(
            "/api/studies",
            json={"title": "Missing Data Policy"},
        )
        study_id = create_resp.json()["id"]

        resp = await client.patch(
            f"/api/studies/{study_id}",
            json={"status": "active"},
        )

        assert resp.status_code == 422
        assert "data handling" in resp.json()["detail"]

    async def test_prolific_study_requires_completion_paths(
        self, client: AsyncClient
    ):
        create_resp = await client.post(
            "/api/studies",
            json={
                "title": "Missing completion paths",
                "research_spec": {
                    "participant_information": {
                        "participation_summary": "Answer questions about a recent work task.",
                        "data_handling_summary": "Transcripts are stored securely.",
                        "data_controller": "Example Research Ltd",
                        "data_recipients_summary": "The authorized research team.",
                        "risks_summary": "Low risk; questions may be skipped.",
                        "compensation_summary": "Payment is described in Prolific.",
                        "researcher_identity": "Example research team",
                        "researcher_contact": "Contact us through Prolific.",
                        "complaints_contact": "Contact the research ethics team.",
                        "data_retention_period": "Responses are kept for 12 months.",
                        "withdrawal_instructions": "Contact the researcher to request deletion.",
                    },
                    "fielding": {"recruitment_source": "prolific"},
                },
            },
        )
        study_id = create_resp.json()["id"]
        resp = await client.patch(
            f"/api/studies/{study_id}", json={"status": "active"}
        )
        assert resp.status_code == 422
        assert "return URLs" in resp.json()["detail"]

    async def test_update_study_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.patch(
            f"/api/studies/{fake_id}",
            json={"title": "Nope"},
        )
        assert resp.status_code == 404

    async def test_delete_study(self, client: AsyncClient):
        # Create
        create_resp = await client.post(
            "/api/studies",
            json={"title": "Delete Me"},
        )
        study_id = create_resp.json()["id"]

        # Delete
        resp = await client.delete(f"/api/studies/{study_id}")
        assert resp.status_code == 204

        # Verify gone
        get_resp = await client.get(f"/api/studies/{study_id}")
        assert get_resp.status_code == 404

    async def test_delete_study_not_found(self, client: AsyncClient):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/api/studies/{fake_id}")
        assert resp.status_code == 404

    async def test_list_studies_returns_created(self, client: AsyncClient):
        await client.post("/api/studies", json={"title": "Study A"})
        await client.post("/api/studies", json={"title": "Study B"})

        resp = await client.get("/api/studies")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        titles = [s["title"] for s in data]
        assert "Study A" in titles
        assert "Study B" in titles
