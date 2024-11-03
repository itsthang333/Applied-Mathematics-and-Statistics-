import math
import numpy as np 
from fractions import Fraction

A = [[1, 3, -2,  0, 2,  0, 0],
     [2, 6, -5, -2, 4, -3, 0],
     [0, 0,  5, 10, 0, 15, 0],
     [2, 6,  0,  8, 4, 18, 0]]

# Reference functions:
def myprint(x, sep=" "):
    if isinstance(x, list) and x:
        if isinstance(x[0], list): # list of list
            m, n = len(x), len(x[0])
            widths = [max(len(str(ai[j])) for ai in x) for j in range(n)]
            rows = [sep.join(format(str(ai[j]), f">{widths[j]}") for j in range(n)) for ai in x]
            print("[" + "\n".join((" [" if i > 0 else "[") + rows[i] + "]" for i in range(m)) + "]")
        else: # list
            print("[" + sep.join(str(e) for e in x) + "]")
    else:
        print(x)
        
def is_zero(x):
    return math.isclose(x, 0, abs_tol=1e-09)

def is_one(v):
    return math.isclose(v, 1)

def to_fraction(x):
    if isinstance(x, list):
        return [to_fraction(e) for e in x] # list comprehension
    else:
        return Fraction(x) # "1/2" --> 1/2

# Gauss Elimination Function
def Gauss_elimination(A, leading1):
    m, n =len(A), len(A[0])

    #Step1: Check if the leftmost column contains all zeros
    def is_column_zero(col_index):
        i = col_index
        for i in range(m):
            if not is_zero(A[i][col_index]):
                return False
        return True

    def is_sub_column_zero(col_index, row_index):
        for i in range(row_index, m):
            if A[i][col_index] !=0:
                return False
        return True
        
    #Step2: Swap rows if it !=0;
    def swap_rows(row_index):
        for i in range (row_index + 1,m):
            if not is_zero(A[i][row_index]) and is_one(A[i][row_index]):
                temp = A[row_index]
                A[row_index] = A[i]
                A[i] = temp
                break

    def swap_rows_3(row_index, col_index):
        best_row = row_index
        best_col = n  

        for i in range(row_index + 1, m):
            for j in range(col_index, n):
                if not is_zero(A[i][j]) and j < best_col:
                    best_row = i
                    best_col = j
                    break

        if best_row != row_index:
            temp_row = A[row_index]
            A[row_index] = A[best_row]
            A[best_row] = temp_row
                
    #Step3: Make leading 1:
    def make_leading_one(row_index, col_index=None):
        if col_index is None:
            col_index = row_index
        leading_coefficient = A[row_index][col_index]
        if leading_coefficient != 0 and not is_one(leading_coefficient):
            scale_factor = 1 / leading_coefficient
            A[row_index] = [element * scale_factor for element in A[row_index]]

    #Step4: Make elements below leading term zero
    def eliminate_below(row_index, col_index=None):
        if col_index is None:
            col_index = row_index
        for i in range(row_index + 1, m):
            if not is_zero(A[row_index][col_index]):
                factor = A[i][col_index] / A[row_index][col_index]
                A[i] = [A[i][j] - factor * A[row_index][j] for j in range(n)]

    #step5: Main loop for Gauss Elimination
    for i in range(min(m, n)):
        if not is_column_zero(i) and not is_sub_column_zero(i,i):
            swap_rows(i)
            if leading1:
                make_leading_one(i)
            eliminate_below(i)
        elif is_sub_column_zero(i,i) :
            if i+1<n and not is_sub_column_zero(i+1,i):
                swap_rows(i - 1)
                if leading1:
                    make_leading_one(i, i+1)
                eliminate_below(i, i+1)
            elif i+1<n and is_sub_column_zero(i+1,i):
                next_column = i + 1
                while next_column<n and is_sub_column_zero(next_column, i) :
                    next_column += 1

                if i+next_column-2<n:
                    swap_rows_3(i,i+next_column-2)
                    if leading1:
                        make_leading_one(i, i+next_column-2)
                    eliminate_below(i, i+next_column-2)

    return A

def back_substitution(R):
    
    # Function to count the number of zero vectors needed
    def count_zero_vectors_needed(R):
        num_rows = len(R)
        num_cols = len(R[0])
        return max(0, num_cols - 1 - num_rows)
    
    # Function to add zero vectors if needed
    def add_zero_vectors(R):
        num_cols = len(R[0])
        num_zero_vectors_needed = count_zero_vectors_needed(R)
        
        if num_zero_vectors_needed == 0:
            return R
        
        # Add zero vectors
        R2 = [row[:] for row in R]  # Copy R to R2
        for _ in range(num_zero_vectors_needed):
            zero_vector = [0] * num_cols
            R2.append(zero_vector)
        
        return R2
    
     # Count zero vectors needed and add them if necessary
    R = add_zero_vectors(R)
    m, n = len(R), len(R[0])
    solutions = [None] * (n - 1)

    for i in range(m - 1, -1, -1):
        row = R[i]
        
        # Check if the row has all zeroes except the last element
        if all(is_zero(row[j]) for j in range(n - 1)):
            if not is_zero(row[-1]):
                return "No solution."
            # Assign free variable
            free_var = None
            for j in range(n - 1):
                if solutions[j] is None:
                    free_var = j
                    break
            if free_var is not None:
                solutions[free_var] = f"x{free_var + 1}"
        
        # Find the pivot column
        pivot_col = None
        for j in range(n - 1):
            if not is_zero(row[j]):
                pivot_col = j
                break
        
        if pivot_col is not None:
            rhs = row[-1]
            equation = []
            for j in range(pivot_col + 1, n - 1):
                if not is_zero(row[j]):
                    if solutions[j] is not None:
                        if isinstance(solutions[j], (int, float, Fraction)):
                            rhs -= row[j] * solutions[j]
                        else:
                            equation.append(f"({-row[j]})*{solutions[j]}")
                    else:
                        equation.append(f"({-row[j]})*x{j+1}")
            if equation:
                equation_str = " - ".join(equation)
                solutions[pivot_col] = f"{rhs} - {equation_str}"
            else:
                solutions[pivot_col] = rhs / row[pivot_col]
        else:
            free_var = None
            for j in range(n - 1):
                if solutions[j] is None:
                    free_var = j
                    break
            if free_var is not None:
                solutions[free_var] = f"x{free_var + 1}"

    # Convert solutions to string and filter out zero terms
    def clean_solution(sol):
        if isinstance(sol, str):
            terms = sol.split(" - ")
            cleaned_terms = [term for term in terms if not any(is_zero(float(f)) for f in term.split('*') if f.replace('.', '', 1).isdigit())]
            return " - ".join(cleaned_terms)
        return str(sol)

    solutions_str = [clean_solution(sol) if sol is not None else "0" for sol in solutions]
    
    return solutions_str

print('\nMatrix after transforming into REF (with leading 1):')
myprint(Gauss_elimination(to_fraction(A), True))

print('\nMatrix after transforming into REF (without leading 1):')
myprint(Gauss_elimination(to_fraction(A), False))

R = Gauss_elimination(to_fraction(A), True)
solutions = back_substitution(R)
print("Solution:")
print(", ".join(solutions))
