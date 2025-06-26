from pydantic import BaseModel, Field
from typing import Optional

class admin_format(BaseModel):
    starting_message: Optional[str] = Field(..., description="(Optional) The starting message for the admin agent to initiate the debate. The Admin introduces the topic and provides a brief summary of the topic.")
    ending_message: Optional[str] = Field(..., description=" (Optional) The ending message for the admin agent to conclude the debate. The Admin provides a summary of the debate and the final decision made by the judge.")

    def format(self) -> str:

        if self.ending_message is None:
            return f"""
                **Admin Agent**    
                
                {self.starting_message}
            """
        
        else:
            return f"""
                **Admin Agent**    

                {self.ending_message}
            """