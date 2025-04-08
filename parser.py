from typing import List, Set, Union

class EinopsError(Exception):
    pass

class ParsedExpression:
    """
    Parses einops pattern expression to apply operations.
    Arguments:
        expression (str): The einops pattern to parse
    Attributes:
        has_ellipsis (bool): Whether the pattern contains an ellipsis
        identifiers (Set[str]): Set of unique dimension names in the pattern
        composition (List[Union[List[str], str]]): Parsed structure of the pattern
        actual_dim_count (int): Number of actual dimensions after considering grouping
    """    
    def __init__(self, expression: str):
        self.has_ellipsis = False
        self.identifiers: Set[str] = set()
        self.composition: List[Union[List[str], str]] = []
        self.actual_dim_count = 0

        # Validate expression is not empty
        if not expression or expression.isspace():
            raise EinopsError("Expression cannot be empty")
        
        # Handle ellipsis first
        if "..." in expression:
            if str.count(expression, "...") > 1:
                raise EinopsError("Multiple ellipsis not allowed in pattern")
            expression = expression.replace("...", "_ellipsis_")
            self.has_ellipsis = True
        
        tokens = []
        current_token = ""
        for char in expression:
            # Check for invalid special characters
            if not (char.isalnum() or char in "() _." or char.isspace()):
                raise EinopsError(f"Invalid character in pattern: '{char}'")

            if char in "() ":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char != " ":
                    tokens.append(char)
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
            
        bracket_group = None
        
        def add_axis_name(name: str):
            if name == "_ellipsis_":
                if bracket_group is not None:
                    raise EinopsError("Ellipsis inside parenthesis not allowed")
                self.composition.append("_ellipsis_")
                return
                
            if not name:
                raise EinopsError("Empty axis name not allowed")  
            elif name[0].isdigit():
                raise EinopsError(f"Axis name cannot start with a number: '{name}'")
            elif not all(c.isalnum() or c == '_' for c in name):
                raise EinopsError(f"Invalid axis name: {name}")
                
            if bracket_group is not None:
                bracket_group.append(name)
            else:
                self.composition.append([name])

            self.identifiers.add(name)
        
        # Process tokens
        for token in tokens:
            if token == "(":
                if bracket_group is not None:
                    raise EinopsError("Nested parentheses not allowed")
                bracket_group = []
            elif token == ")":
                if bracket_group is None:
                    raise EinopsError("Unmatched closing parenthesis")
                if not bracket_group:
                    raise EinopsError("Empty parentheses not allowed")
                self.composition.append(bracket_group)
                self.actual_dim_count += 1 
                bracket_group = None
            else:
                add_axis_name(token)
                if bracket_group is None:
                    self.actual_dim_count += 1
                
        if bracket_group is not None:
            raise EinopsError("Unclosed parenthesis")