from llama_index.core import PromptTemplate
from typing import Optional
from src.data import Dependency


class _Prompts:
    _query_prompt: PromptTemplate = PromptTemplate(
        "Information about both configuration options, such as their descriptions or prior usages are below:\n\n"
        "{context_str}\n\n"
        "Given the context information, perform the following task:\n\n"
        "{task_str}\n\n"
        "{format_str}"
    )

    _advanced_query_prompt: PromptTemplate = PromptTemplate(
        "Information about both configuration options, including their descriptions, prior usages, and examples of similar dependencies are provided below.\n"
        "The provided information comes from various sources, such as manuals, Stack Overflow posts, GitHub repositories, and web search results.\n"
        "Note that not all the provided information may be relevant for validating the dependency.\n"
        "Consider only the information that is relevant for validating the dependency, and disregard the rest."
        "{context_str}\n"
        "Additionally, here are some examples on how similar dependencies are evaluated:\n\n"
        "{shot_str}\n\n"
        "Given the information and examples, perform the following task:\n\n"
        "{task_str}\n\n"
        "{format_str}"
    )

    _system_prompt : Optional[PromptTemplate] = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project_name}'.\n" 
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
        "{dependency_str}"
    )

    _advanced_system_prompt : Optional[PromptTemplate] = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project_name}'.\n\n" 
        "{project_info}\n\n"
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n"
        "{dependency_str}"
    )
    
    _task_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} "
        "depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
    )

    _retrieval_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Dependency between {nameA} in {technologyA} with value {valueA} and {nameB} in {technologyB} with value {valueB}"
    )

    _dependency_prompt: Optional[PromptTemplate] = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly.\n"
        "Inconsistencies in these configuration values can lead to configuration errors.\n"
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values."
    )

    _advanced_dependency_prompt: Optional[PromptTemplate] = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly.\n"
        "Inconsistencies in these configuration values can lead to configuration errors.\n"
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values.\n"
        "If the values of configuration options are identical merely to ensure consistency within a software project, the options are not considered dependent."
    ) 

    _format_str: Optional[PromptTemplate] = PromptTemplate(
        "Respond in a JSON format as shown below:\n"
        "{{\n"
        "\t“plan”: string, // Write down a step-by-step plan on how to solve the task given the information and examples of similar dependencies above.\n"
        "\t“rationale”: string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.\n"
        "\t“isDependency”: boolean // True if a dependency exists, or False otherwise.\n"
        "}}"
    )

    def get_query_str(self, context_str: str, task_str: str, shot_str: Optional[str], advanced: bool = False) -> str:
        """Get formatted query prompt."""
        if advanced:
            return self._advanced_query_prompt.format(
                context_str=context_str,
                task_str=task_str,
                shots_str=shot_str,
                format_str=self.get_format_prompt()
            )

        return self._query_prompt.format(
            context_str=context_str,
            task_str=task_str,
            format_str=self.get_format_prompt()
        )

    def get_system_str(self, project_name: str, project_info: Optional[str] = None, advanced: bool = False) -> str:
        """Get formatted system prompt."""
        if advanced:
            return self._advanced_system_prompt.format(
                project_name=project_name,
                project_info=project_info,
                dependency_str=self.get_dependency_definition_str(advanced=advanced)
            )

        return self._system_prompt.format(
            project_name=project_name,
            dependency_str=self.get_dependency_definition_str(advanced=advanced)
        )
        
    def get_task_str(self, dependency: Dependency) -> str:
        """Get formatted system task prompt."""
        return self._task_prompt.format(
            nameA=dependency.option_name,
            typeA=dependency.option_type,
            valueA=dependency.option_value,
            fileA=dependency.option_file,
            technologyA=dependency.option_technology,
            nameB=dependency.dependent_option_name,
            typeB=dependency.dependent_option_type,
            valueB=dependency.dependent_option_value,
            fileB=dependency.dependent_option_file,
            technologyB=dependency.dependent_option_technology
        )
        
    def get_dependency_definition_str(self, advanced: bool) -> str:
        """Get formatted dependency prompt."""
        if advanced:
            return self._advanced_dependency_prompt.format()
        return self._dependency_prompt.format()
    
    def get_retrieval_prompt(self, dependency: Dependency) -> str:
        """Get formatted retrieval prompt."""
        return self._retrieval_prompt.format(
            nameA=dependency.option_name,
            technologyA=dependency.option_technology,
            valueA=dependency.option_value,
            nameB=dependency.dependent_option_name,
            technologyB=dependency.dependent_option_technology,
            valueB=dependency.dependent_option_value
        )
    

    def get_format_prompt(self) -> str:
        return self._format_str.format()

# Singelton
Prompts = _Prompts()


