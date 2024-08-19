"""
    Documentation
"""

import string

########################################
# Constants
########################################
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS
KEYWORDS = [
    'VAR',
    'AND',
    'OR',
    'NOT',
    'IF', 
    'THEN',
    'ELIF',
    'ELSE'
]

########################################
# Error
########################################

def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
    
    # Generate each line
    line_count = pos_end.line_number - pos_start.line_number + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.column if i == 0 else 0
        col_end = pos_end.column if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '')

class Error:
    """
    Documentation
    """
    def __init__(self, pos_start, pos_end, error_name, details):
        self.error_name = error_name
        self.details = details
        # self.column = column
        # self.text = text
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        """
        Documentation
        """
        result = "\n\nERROR:\n"
        # result += self.pos_start.file_text[int(self.pos_start.column*self.pos_start.line_number):int((self.pos_start.column+1)*(self.pos_start.line_number+2)+1)] + '\n'
        # result += " "*(self.pos_start.column) + "^"
        result += f'\n{self.error_name} : {self.details}'
        result += f' \nFile {self.pos_start.file_name}, line {self.pos_start.line_number + 1}, at column : {self.pos_start.column + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.file_text, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    """
    Documentation
    """
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class ExpectedCharError(Error):
    """
    Documentation
    """
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
    """
    Documentation
    """
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Syntax', details)
        
class RunTimeError(Error):
    """
    Documentation
    """
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Run-time Error', details)
        self.context = context
    
    def as_string(self):
        """
        Documentation
        """
        result = self.generate_traceback()
        result += f'\n{self.error_name} : {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.file_text, self.pos_start, self.pos_end)
        return result
    
    def generate_traceback(self):
        result = ''
        ctx = self.context
        pos = self.pos_start
        
        while(ctx):
            result = f' File {pos.file_name}, line {pos.line_number + 1}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent
        return 'Traceback (most recent call last:)\n' + result
        
########################################
# Position
########################################

class Position:
    """
    Documentation
    """
    def __init__(self, idx, line_number, column, file_name, file_text):
        self.idx = idx
        self.line_number = line_number
        self.column = column
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, current_char = None):
        """
        Documentation
        """
        self.idx += 1
        self.column += 1

        if current_char == '\n':
            self.line_number += 1
            self.column = 0

        return self

    def copy(self):
        """
        Documentation
        """
        return Position(self.idx, self.line_number, self.column, self.file_name, self.file_text)

########################################
# Token
########################################

TOKEN_TYPE_INT = 'INT'
TOKEN_TYPE_FLOAT = 'FLOAT'
TOKEN_TYPE_PLUS = 'PLUS'
TOKEN_TYPE_MINUS = 'MINUS'
TOKEN_TYPE_MULTIPLICATION = 'MULTIPLICATION'
TOKEN_TYPE_DIVISION = 'DIVISION'
TOKEN_TYPE_RIGHT_PARENTHESIS = 'RIGHT_PARENTHESIS'
TOKEN_TYPE_LEFT_PARENTHESIS = 'LEFT_PARENTHESIS'
TOKEN_TYPE_POWER = 'POWER'
TOKEN_TYPE_EOF = 'EOF'
TOKEN_TYPE_KEYWORD = 'KEYWORD'
TOKEN_TYPE_IDENTIFIER = 'IDENTIFIER'
TOKEN_TYPE_EQ = 'EQUAL'
TOKEN_TYPE_EE = 'EE' 
TOKEN_TYPE_NE = 'NE' 
TOKEN_TYPE_LT = 'LT' 
TOKEN_TYPE_GT = 'GT' 
TOKEN_TYPE_LTE = 'LTE' 
TOKEN_TYPE_GTE = 'GTE'
class Token:
    """
    Documentation
    """
    def __init__(self, type_, value = None, pos_start=None , pos_end=None):
        self.type = type_
        self.value = value
        
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end
            

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        else:
            return f'{self.type}'
    
    def matches(self, type_, value):
        return self.type == type_ and self.value == value

########################################
# Lexer
########################################

class Lexer:
    """
    Documentation
    """
    def __init__(self, file_name, text):
        self.text = text
        self.pos = Position(-1, 0, -1, file_name, text)
        self.current_char = None
        self.advance()

    def advance(self):
        """
        Documentation
        """
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        """
        Documentation
        """
        tokens = []

        while self.current_char is not None:
            if self.current_char in ' ':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(TOKEN_TYPE_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TOKEN_TYPE_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                # print(self.text[self.pos.idx], self.text[self.pos.idx+1])
                if self.pos.idx + 1 < len(self.text) and self.text[self.pos.idx + 1] == '*':
                    tokens.append(Token(TOKEN_TYPE_POWER, pos_start=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(TOKEN_TYPE_MULTIPLICATION, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TOKEN_TYPE_DIVISION, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TOKEN_TYPE_POWER, pos_start=self.pos))
                self.advance()
            # elif self.current_char == '=':
            #     tokens.append(Token(TOKEN_TYPE_EQ, pos_start=self.pos))
            #     self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TOKEN_TYPE_LEFT_PARENTHESIS, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TOKEN_TYPE_RIGHT_PARENTHESIS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                token, error = self.make_not_equals()
                if error: return [], error
                tokens.append(token) 
            elif self.current_char == '=':
                # tokens.append(Token(TOKEN_TYPE_RIGHT_PARENTHESIS, pos_start=self.pos))
                # self.advance()
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                # return [], IllegalCharError("'" + char + "'")
                # # returning no tokens and error if there is an illegal character
                return tokens, IllegalCharError(pos_start, self.pos, "'" + char + "'")
            # returning no tokens and error if there is an illegal character
            
        tokens.append(Token(TOKEN_TYPE_EOF, pos_start=self.pos))
        return tokens, None # returning tokens and no error

    def make_number(self):
        """
        Documentation
        """
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                else:
                    dot_count += 1
                    num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TOKEN_TYPE_INT, int(num_str), pos_start, self.pos)
        elif dot_count == 1:
            return Token(TOKEN_TYPE_FLOAT, float(num_str), pos_start, self.pos)
    
    def make_identifier(self):
        """_summary_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        id_str = ''
        pos_start = self.pos.copy()
        
        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()
        
        token_type = TOKEN_TYPE_KEYWORD if id_str in KEYWORDS else TOKEN_TYPE_IDENTIFIER
        return Token(token_type, id_str, pos_start, self.pos)
    
    
    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TOKEN_TYPE_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
    
    def make_equals(self):
        token_type = TOKEN_TYPE_EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            token_type = TOKEN_TYPE_EE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        token_type = TOKEN_TYPE_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            token_type = TOKEN_TYPE_LTE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        token_type = TOKEN_TYPE_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            token_type = TOKEN_TYPE_GTE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

        
########################################
# Nodes
########################################

class NumberNode:
    def __init__(self, token):
        self.token = token
        
        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end
    
    def __repr__(self):
        return f'{self.token}'

class BinOpNode:
    def __init__(self, left_node, op_token, right_node):
        self.left_node = left_node
        self.op_token = op_token
        self.right_node = right_node
        
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_token}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op_token, node):
        self.op_token = op_token
        self.node = node
        
        self.pos_start = self.op_token.pos_start
        self.pos_end = self.node.pos_end
        
    def __repr__(self):
        return f'({self.op_token}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0]).pos_end

class VarAccessNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end

class VarAssignNode:
    def __init__(self, var_name_token, value_node):
        self.var_name_token = var_name_token
        self.value_node = value_node
        
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end

########################################
# Parse Result
########################################
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
    
    # def register(self, res):
    #     if isinstance(res, ParseResult):
    #         if res.error:
    #             self.error = res.error
    #         return res.node
    #     return res

    def register_advancement(self):
        self.advance_count += 1

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self
########################################
# Parser
########################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_idx = -1
        self.advance()
    
    def advance(self):
        self.token_idx += 1
        if self.token_idx < len(self.tokens):
            self.current_token = self.tokens[self.token_idx]
        return self.current_token

    ###################################
    
    def parse(self):
        res = self.expr()
        # print(res.node, res.error)
        if not res.error and self.current_token.type != TOKEN_TYPE_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
            ))
        return res
    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_token.matches(TOKEN_TYPE_KEYWORD, 'IF'):
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f"Expected 'IF'"))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error:
            return res

        if not self.current_token.matches(TOKEN_TYPE_KEYWORD, 'THEN'):
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f"Expected 'THEN'"))

        res.register_advancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error:
            return res
        cases.append((condition, expr))

        while self.current_token.matches(TOKEN_TYPE_KEYWORD, 'ELIF'):
            res.register_advancement()
            self.advance()

            condition = res.register(self.expr())
            if res.error:
                return res

            if not self.current_token.matches(TOKEN_TYPE_KEYWORD, 'THEN'):
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f"Expected 'THEN'"))

            res.register_advancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error:
                return res
            cases.append((condition, expr))

        if self.current_token.matches(TOKEN_TYPE_KEYWORD, 'ELSE'):
            res.register_advancement()
            self.advance()

            else_case = res.register(self.expr())
            if res.error:
                return res

        return res.success(IfNode(cases, else_case))
       
        
        
    def atom(self):
        res = ParseResult()
        token = self.current_token
        
        if token.type in (TOKEN_TYPE_INT , TOKEN_TYPE_FLOAT): # int / float
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(token))
        elif token.type == TOKEN_TYPE_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(token))            
        elif token.type == TOKEN_TYPE_LEFT_PARENTHESIS: # (
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: # Error if only ( and no closing ')' or any other syntax error
                return res
            if self.current_token.type == TOKEN_TYPE_RIGHT_PARENTHESIS: # if closing ')' is found without any errors
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSyntaxError(
                        token.pos_start, 
                        token.pos_end, 
                        "Expected ')'"
                    )
                )
        elif token.matches(TOKEN_TYPE_KEYWORD, 'IF'):
            if_expr = res.register(self.if_expr())
            # print("12123", if_expr)
            if res.error:
                return res
            return res.success(if_expr)
        return res.failure(InvalidSyntaxError(token.pos_start, token.pos_end, "Expected int, float, identifier, '+', '-' or '('"))
    
    def power(self):
        """
        Parses the power operation.

        The power operation has the form: <atom> ** <factor>.
        It is a binary operation that takes two operands of the same type.

        Returns:
            ParseResult: The result of parsing the power operation.
        """
        # Call the bin_op method with the appropriate arguments.
        # The atom is the left operand, the TOKEN_TYPE_POWER is the operation,
        # and the factor is the right operand.
        return self.bin_op(self.atom, (TOKEN_TYPE_POWER, ), self.factor)
    
    def factor(self):
        res = ParseResult()
        token = self.current_token
        # print(token, token.type)
        
        if token.type in (TOKEN_TYPE_PLUS, TOKEN_TYPE_MINUS): #  +a / -a
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(token, factor))
        
        # else:
        return self.power()
    
    def term(self):
        return self.bin_op(self.factor, (TOKEN_TYPE_MULTIPLICATION, TOKEN_TYPE_DIVISION))
    
    def arith_expr(self):
        return self.bin_op(self.term, (TOKEN_TYPE_PLUS, TOKEN_TYPE_MINUS))

    def comp_expr(self):
        res = ParseResult()

        if self.current_token.matches(TOKEN_TYPE_KEYWORD, 'NOT'):
            op_token = self.current_token
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(op_token, node))
        
        node = res.register(self.bin_op(self.arith_expr, (TOKEN_TYPE_EE, TOKEN_TYPE_NE, TOKEN_TYPE_LT, TOKEN_TYPE_GT, TOKEN_TYPE_LTE, TOKEN_TYPE_GTE)))
        
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected int, float, identifier, '+', '-', '(' or 'NOT'"
            ))

        return res.success(node)
        
    def expr(self):
        res = ParseResult()
        
        # VAR abcd = EXPRESSION[expr]
        if self.current_token.matches(TOKEN_TYPE_KEYWORD, 'VAR'):
            res.register_advancement()
            self.advance()

            if self.current_token.type != TOKEN_TYPE_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected identifier"
                ))
            
            var_name =  self.current_token
            res.register_advancement()
            self.advance()
            
            if self.current_token.type != TOKEN_TYPE_EQ:
                res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected '='"
                ))
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(VarAssignNode(var_name, expr))
        # return self.bin_op(self.term, (TOKEN_TYPE_PLUS, TOKEN_TYPE_MINUS))
        node = res.register(self.bin_op(self.comp_expr, ((TOKEN_TYPE_KEYWORD, 'AND'), (TOKEN_TYPE_KEYWORD, 'OR'))))
        # node = res.register(self.bin_op(self.comp_expr, ((TOKEN_TYPE_KEYWORD, 'AND'), (TOKEN_TYPE_KEYWORD, 'OR'))))
        

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                "Expected 'VAR', int, float, identifier, '+', '-' or '('"
            ))

        return res.success(node)
    
    ###################################
    
    def bin_op(self, func_a, ops, func_b = None): # func = terms/factors, ops = +/*/division/-
        if func_b == None:
            func_b = func_a
            
        res = ParseResult()
        left = res.register(func_a())
        if res.error:
            return res
            
        while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
            op_token = self.current_token
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error:
                return res
            left = BinOpNode(left, op_token, right)
        
        return res.success(left)

########################################
# Runtime Result
########################################
class RuntimeResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self
        
    
########################################
# Values
########################################

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
        
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self
    
    def added_by(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RunTimeError(other.pos_start, other.pos_end, 'Division by zero is undefined and cannot be allowed', self.context)
            else:
                return Number(self.value / other.value).set_context(self.context), None
    
    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value  ** other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None
        
    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def is_true(self):
        return self.value != 0
        
    def __repr__(self):
        return f'{self.value}'

########################################
# Context
########################################

class Context:
    def __init__(self, display_name, parent = None, parent_entry_pos = None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None
        
########################################
# Symbol Table
########################################
class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]
    
########################################
# Interpreter
########################################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        # print(method_name)
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')
    
    #######################
    
    def visit_NumberNode(self, node, context):
        # print('Found number node')
        return RuntimeResult().success(
            Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
            ) # Because every number is successful
    
    def visit_VarAccessNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RunTimeError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name_token.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    
    def visit_BinOpNode(self, node, context):
        res = RuntimeResult()
        # print('Found binary operator node', node, node.left_node, node.right_node)
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res
        
        if node.op_token.type == TOKEN_TYPE_PLUS:
            result, error = left.added_by(right)
        if node.op_token.type == TOKEN_TYPE_MINUS:
            result, error = left.subbed_by(right)
        if node.op_token.type == TOKEN_TYPE_MULTIPLICATION:
            result, error = left.multed_by(right)
        if node.op_token.type == TOKEN_TYPE_DIVISION:
            result, error = left.dived_by(right)
        if node.op_token.type == TOKEN_TYPE_POWER:
            result, error = left.powed_by(right)
        # else:
        #     print("Couldn't", node.op_token.value,  node.op_token.type)
        elif node.op_token.type == TOKEN_TYPE_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_token.type == TOKEN_TYPE_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_token.type == TOKEN_TYPE_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_token.type == TOKEN_TYPE_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_token.type == TOKEN_TYPE_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_token.type == TOKEN_TYPE_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_token.matches(TOKEN_TYPE_KEYWORD, 'AND'):
            result, error = left.anded_by(right)
        elif node.op_token.matches(TOKEN_TYPE_KEYWORD, 'OR'):
            result, error = left.ored_by(right)
        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
    
    def visit_UnaryOpNode(self, node, context):
        # print('Found unary operator node', node, node.node)
        res = RuntimeResult()
        
        error = None
        number = res.register(self.visit(node.node, context))
        
        if res.error:
            return res
        
        if node.op_token.type == TOKEN_TYPE_MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_token.matches(TOKEN_TYPE_KEYWORD, 'NOT'):
            number, error = number.notted()
        
        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))
    
    def visit_IfNode(self, node, context):
        res = RuntimeResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error:
                return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error:
                    return res
                return res.success(expr_value)

        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(else_value)

        return res.success(None)
    
########################################
# Run
########################################
global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0))
global_symbol_table.set("KAHI_NAHI", Number(0))
global_symbol_table.set("KHOT", Number(0))
global_symbol_table.set("TRUE", Number(1))
global_symbol_table.set("KHAR", Number(1))

def run(file_name, text):
    """
    Documentation
    """
    # Generating tokens
    lexer = Lexer(file_name, text)
    tokens, error = lexer.make_tokens()
    if error:
        return tokens, error
    
    # Generating AST (Abstract Syntax Tree)
    parser = Parser(tokens)
    ast = parser.parse()
    
    if ast.error:
        return None, ast.error
    
    # Running interpreter
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)
        
    return result.value, result.error