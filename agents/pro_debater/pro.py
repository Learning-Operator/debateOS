from pydantic import BaseModel, Field
from typing import Optional


class pro_format(BaseModel):
    Argument: str = Field(..., description="The Argument by the pro debater agent. The argument MUST be written in paragraph form")
   
    def format(self) -> str:
        return f"""
            **Pro Debater Agent**
            {self.Argument},
        """