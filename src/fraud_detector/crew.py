# src/fraud_detector/crew.py
from typing import List
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os

class RiskAssesment(BaseModel):
    """Risk Assesment model for the Fraud Detector crew."""
    risk_score: float = Field(..., description="The risk score of the lead between 0 - 10")
    risk_summary: str = Field(..., description="The summary of the risk of the lead")
    risk_factors: List[str] = Field(..., description="The factors that contribute to the risk score")

@CrewBase
class FraudDetectorCrew:
    """Fraud Detection crew configured to identify and assess potential fraud activities."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _get_llm(self):
        """Creates an LLM instance configured for Groq."""
        return LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",  # Changed to model with higher TPM
            temperature=0.5,
            max_tokens=4096,
            request_timeout=300
        )

    def _create_agent(self, role_config):
        """Helper method to create an agent with consistent configuration."""
        return Agent(
            role=role_config["role"],
            goal=role_config["goal"],
            backstory=role_config["backstory"],
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            llm=self._get_llm(),
            verbose=True
        )

    @agent
    def financial_forensics_analyst(self) -> Agent:
        return self._create_agent(self.agents_config["financial_forensics_analyst"])

    @agent
    def compliance_officer(self) -> Agent:
        return self._create_agent(self.agents_config["compliance_officer"])

    @agent
    def risk_assessment_analyst(self) -> Agent:
        return self._create_agent(self.agents_config["risk_assessment_analyst"])

    @task
    def financial_forensics_task(self) -> Task:
        task_config = self.tasks_config["financial_forensics_analyst_task"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.financial_forensics_analyst()
        )

    @task
    def compliance_task(self) -> Task:
        task_config = self.tasks_config["compliance_officer_task"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.compliance_officer()
        )

    @task
    def risk_assessment_task(self) -> Task:
        task_config = self.tasks_config["risk_assessment_analyst_task"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.risk_assessment_analyst(),
            output_json=RiskAssesment
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Fraud Detection crew"""
        return Crew(
            agents=[
                self.financial_forensics_analyst(),
                self.compliance_officer(),
                self.risk_assessment_analyst()
            ],
            tasks=[
                self.financial_forensics_task(),
                self.compliance_task(),
                self.risk_assessment_task()
            ],
            process=Process.sequential,
            verbose=True
        )