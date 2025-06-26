from pydantic import BaseModel, Field
from typing import Optional, Literal


class judge_format(BaseModel):
    summary: str = Field(..., description="The summary of the debate provided by the judge agent.")
    Decision: Literal['The Pro side won the debate', 'The Con side won the debate' ] = Field(..., description="The decision to which side of the argument is more convincing, either 'pro' or 'con'.")
    Reasoning: str = Field(..., description="The reasoning provided by the judge to why they made their decision.")

    def format(self) -> str:

        if self.response == None:
            return f"""
                **Judge Agent**

                {self.summary}

                {self.Decision},

                {self.Reasoning}

            """
        