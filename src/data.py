from dataclasses import dataclass
from typing import Optional
    
    
@dataclass
class Dependency:
    project: Optional[str] = None
    dependency_category: Optional[str] = None
    option_name: Optional[str] = None
    option_file: Optional[str] = None 
    option_value: Optional[str] = None
    option_type: Optional[str] = None
    option_technology: Optional[str] = None
    dependent_option_name: Optional[str] = None
    dependent_option_value: Optional[str] = None
    dependent_option_type: Optional[str] = None
    dependent_option_file: Optional[str] = None 
    dependent_option_technology: Optional[str] = None
    rating: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None


    def to_dict(self):
        """Convert dependency into a dictionary."""
        return {
            "project": self.project,
            "dependency_category": self.dependency_category,
            "option_name": self.option_name,
            "option_file": self.option_file,
            "option_value": self.option_value,
            "option_type": self.option_type,
            "option_technology": self.option_technology,
            "dependent_option_name": self.dependent_option_name,
            "dependent_option_value": self.dependent_option_value,
            "dependent_option_file": self.dependent_option_file,
            "dependent_option_type": self.dependent_option_type,
            "dependent_option_technology": self.dependent_option_technology,
            "rating": self.rating,
            "category": self.category,
            "sub_category": self.sub_category
        }