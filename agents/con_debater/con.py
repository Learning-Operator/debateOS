from pydantic import BaseModel, Field
from typing import Optional


class con_format(BaseModel):
    argument: str = Field(..., description="The Argument by the pro debater agent. The argument MUST be written in paragraph form")

    def format(self) -> str:

        return f"""
            **Con Debater Agent**
            {self.argument}
        """