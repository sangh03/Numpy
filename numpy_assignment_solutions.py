"""
===============================================================================
NumPy Practice Assignment - Complete Python Script
===============================================================================
Author: GitHub Repository Solutions
Topic: Advanced Array Operations, Linear Algebra, Broadcasting, & Data Analysis
Language: Python 3 (NumPy)
"""

import numpy as np

# Set random seed for consistent, reproducible results
np.random.seed(42)


# =============================================================================
# Question 1: Array Creation, Reshaping and Indexing
# =============================================================================
# Problem Statement:
# a) Create a 1D NumPy array named arr1 containing integers from 0 to 59 inclusive.
# b) Reshape arr1 into a 2D array named arr2 with 6 rows and 10 columns.
# c) Reshape arr2 into a 3D array named arr3 with shape (3, 4, 5).
# d) Change element at 2nd row, 5th column of arr2 to -25. Check arr1 and arr3.
# e) Extract slices: first 2 rows, last 3 columns, bottom-right 3x4 submatrix, every second column.
# f) Reverse the row order, and reverse both rows and columns.
# =============================================================================

print("=" * 70)
print("QUESTION 1: Array Creation, Reshaping and Indexing")
print("=" * 70)

# a) Create a 1D array from 0 to 59
arr1 = np.arange(0, 60)
print("\na) arr1 (1D Array):")
print(arr1)

# b) Reshape into a 2D array (6 rows, 10 columns)
arr2 = arr1.reshape(6, 10)
print("\nb) arr2 (2D Array - 6x10):")
print(arr2)

# c) Reshape into a 3D array (shape: 3, 4, 5)
arr3 = arr2.reshape(3, 4, 5)
print("\nc) arr3 Shape:", arr3.shape)

# d) Modify element at second row, fifth column of arr2 (Index: row 1, column 4)
arr2[1, 4] = -25
print("\nd) Modified arr2[1, 4] = -25")
print("   Value in arr1 at index 14:", arr1[14])
print("   Value in arr3 at position (0, 2, 4):", arr3[0, 2, 4])
# Explanation: Modifying arr2 modifies arr1 and arr3 because .reshape() creates 
# a VIEW that shares the underlying memory buffer, not a independent copy.

# e) Slicing operations
print("\ne) Slicing Operations:")
print("   (i) First two rows:\n", arr2[:2, :])
print("   (ii) Last three columns:\n", arr2[:, -3:])
print("   (iii) Bottom-right 3x4 submatrix:\n", arr2[-3:, -4:])
print("   (iv) Every second column:\n", arr2[:, ::2])

# f) Reversing array
print("\nf) Array Reversals:")
print("   Reverse row order:\n", arr2[::-1, :])
print("   Reverse both rows and columns:\n", arr2[::-1, ::-1])


# =============================================================================
# Question 2: NumPy Aggregation and Axis Operations
# =============================================================================
# Problem Statement:
# a) Create a 4 x 6 matrix containing integers from 1 to 24.
# b) Compute sum of all elements.
# c) Compute row-wise sum (axis=1) and column-wise sum (axis=0).
# d) Compute column-wise mean, maximum, and minimum.
# e) Create 3D array (2, 3, 4) and compute sum across axis=0, axis=1, axis=2.
# f) Explain what happens to a dimension when an axis is used in aggregation.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 2: NumPy Aggregation and Axis Operations")
print("=" * 70)

# a) Create a 4x6 matrix from 1 to 24
matrix2 = np.arange(1, 25).reshape(4, 6)
print("\na) Matrix (4x6):\n", matrix2)

# b) Total sum
print("\nb) Total Sum:", matrix2.sum())

# c) Row-wise and column-wise sums
print("\nc) Row-wise Sum (axis=1):", matrix2.sum(axis=1))
print("   Column-wise Sum (axis=0):", matrix2.sum(axis=0))

# d) Column-wise statistics
print("\nd) Column-wise Mean:", matrix2.mean(axis=0))
print("   Column-wise Maximum:", matrix2.max(axis=0))
print("   Column-wise Minimum:", matrix2.min(axis=0))

# e) 3D Array Aggregations
arr3d_2 = np.arange(1, 25).reshape(2, 3, 4)
print("\ne) 3D Array Sums across Axes:")
print("   Sum over axis=0 (Collapses layers, Result shape: 3x4):\n", arr3d_2.sum(axis=0))
print("   Sum over axis=1 (Collapses rows, Result shape: 2x4):\n", arr3d_2.sum(axis=1))
print("   Sum over axis=2 (Collapses columns, Result shape: 2x3):\n", arr3d_2.sum(axis=2))

# f) Theoretical Explanation:
# When an axis `k` is specified in an aggregation, NumPy operates ALONG that axis,
# collapsing (removing) dimension `k` from the resulting shape.


# =============================================================================
# Question 3: Broadcasting with Vectors and Matrices
# =============================================================================
# Problem Statement:
# a) Create a vector x containing integers 1 through 8.
# b) Form an 8x8 matrix A using broadcasting where A[i, j] = x[i] + x[j].
# c) Form an 8x8 matrix B where B[i, j] = x[i] * x[j].
# d) Add row vector [10..80] to every row of A.
# e) Subtract column vector [[10]..[80]] from every column of A.
# f) Multiply a (5, 4) matrix column-wise with a vector of length 4.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 3: Broadcasting with Vectors and Matrices")
print("=" * 70)

# a) 1D vector x
x3 = np.arange(1, 9)
print("\na) Vector x:", x3)

# b) Matrix A where A[i, j] = x[i] + x[j]
A3 = x3.reshape(8, 1) + x3
print("\nb) Matrix A (8x8 Sum Matrix):\n", A3)

# c) Matrix B where B[i, j] = x[i] * x[j]
B3 = x3.reshape(8, 1) * x3
print("\nc) Matrix B (8x8 Outer Multiplication Table):\n", B3)

# d) Add row vector to every row of A
row_vec3 = np.array([10, 20, 30, 40, 50, 60, 70, 80])
A_row_added = A3 + row_vec3
print("\nd) Matrix A + Row Vector:\n", A_row_added)

# e) Subtract column vector from every column of A
A_col_subtracted = A3 - row_vec3.reshape(8, 1)
print("\ne) Matrix A - Column Vector:\n", A_col_subtracted)

# f) 5x4 Matrix multiplied column-wise by a vector of length 4
mat_5x4 = np.arange(1, 21).reshape(5, 4)
vec_len_4 = np.array([2, 3, 4, 5])
result_3f = mat_5x4 * vec_len_4
print("\nf) (5, 4) Matrix * Vector of length 4:\n", result_3f)


# =============================================================================
# Question 4: Random Data and Descriptive Statistics
# =============================================================================
# Problem Statement:
# a) Generate a random array dataset with shape (60, 4) using values from uniform distribution.
# b) Compute overall mean and standard deviation.
# c) Compute column-wise mean and standard deviation.
# d) Find minimum and maximum of every column.
# e) Compute row-wise mean.
# f) Find row indices where row mean > overall mean.
# g) Difference between axis=0 and axis=1 for this dataset.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 4: Random Data and Descriptive Statistics")
print("=" * 70)

# a) Generate (60, 4) dataset
dataset4 = np.random.rand(60, 4)

# b) Overall statistics
print(f"\nb) Overall Mean: {dataset4.mean():.4f}")
print(f"   Overall Std Dev: {dataset4.std():.4f}")

# c) Column-wise statistics
print("\nc) Column-wise Mean:", np.round(dataset4.mean(axis=0), 4))
print("   Column-wise Std Dev:", np.round(dataset4.std(axis=0), 4))

# d) Column min and max
print("\nd) Column Minimums:", np.round(dataset4.min(axis=0), 4))
print("   Column Maximums:", np.round(dataset4.max(axis=0), 4))

# e) Row-wise mean
row_means4 = dataset4.mean(axis=1)

# f) Row indices with mean greater than overall mean
indices_above_mean = np.where(row_means4 > dataset4.mean())[0]
print("\nf) Row indices with row_mean > overall_mean:\n", indices_above_mean)

# g) Explanation of axis=0 vs axis=1 for (60, 4) dataset:
# - axis=0 operates vertically down 60 rows yielding a 1D array of shape (4,).
# - axis=1 operates horizontally across 4 columns yielding a 1D array of shape (60,).


# =============================================================================
# Question 5: Standardization and Normalization
# =============================================================================
# Problem Statement:
# a) Create a dataset with 40 observations and 5 features with values between 0 and 100.
# b) Compute mean and std of each feature.
# c & d) Standardize dataset: Z = (X - mean) / std.
# e) Verify standardized dataset has column mean ≈ 0 and std ≈ 1.
# f) Normalize dataset using Min-Max scaling: X_norm = (X - min) / (max - min).
# g) Verify normalized values lie between 0 and 1.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 5: Standardization and Normalization")
print("=" * 70)

# a) Dataset with 40 observations and 5 features
data5 = np.random.rand(40, 5) * 100

# b) Feature mean and standard deviation
mean5 = data5.mean(axis=0)
std5 = data5.std(axis=0)

# c & d) Standardization (Z-score normalization)
z_standardized = (data5 - mean5) / std5

# e) Verification of Standardization
print("\ne) Standardized Column Means (Target: 0):", np.round(z_standardized.mean(axis=0), 6))
print("   Standardized Column Stds  (Target: 1):", np.round(z_standardized.std(axis=0), 6))

# f) Min-Max Normalization
min5 = data5.min(axis=0)
max5 = data5.max(axis=0)
minmax_normalized = (data5 - min5) / (max5 - min5)

# g) Verification of Min-Max Normalization
print("\ng) Normalized Column Minimums (Target: 0):", np.round(minmax_normalized.min(axis=0), 6))
print("   Normalized Column Maximums (Target: 1):", np.round(minmax_normalized.max(axis=0), 6))


# =============================================================================
# Question 6: Boolean Indexing and Conditional Selection
# =============================================================================
# Problem Statement:
# a) Generate 30 random integer scores between 0 and 100.
# b) Extract scores >= 60.
# c) Extract scores between 40 and 80 inclusive.
# d) Count scores < 35.
# e) Replace all scores < 40 with 0.
# f) Convert scores into binary representation: 1 if score >= 75 else 0.
# g) Find indices of highest 5 scores.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 6: Boolean Indexing and Conditional Selection")
print("=" * 70)

# a) Generate 30 integer scores
scores = np.random.randint(0, 101, 30)
print("\na) Original Scores:\n", scores)

# b) Extract scores >= 60
print("\nb) Scores >= 60:\n", scores[scores >= 60])

# c) Extract scores between 40 and 80 (inclusive)
print("\nc) Scores between 40 and 80 (inclusive):\n", scores[(scores >= 40) & (scores <= 80)])

# d) Count scores < 35
print("\nd) Count of scores < 35:", np.sum(scores < 35))

# e) Replace score < 40 with 0
scores_replaced = scores.copy()
scores_replaced[scores_replaced < 40] = 0
print("\ne) Scores after replacing < 40 with 0:\n", scores_replaced)

# f) Binary conversion: 1 if score >= 75 else 0
binary_scores = np.where(scores >= 75, 1, 0)
print("\nf) Binary Scores (1 if score >= 75 else 0):\n", binary_scores)

# g) Indices of highest 5 scores
top_5_indices = np.argsort(scores)[-5:][::-1]
print("\ng) Indices of top 5 scores:", top_5_indices)
print("   Top 5 scores:", scores[top_5_indices])


# =============================================================================
# Question 7: Matrix Operations and Linear Algebra
# =============================================================================
# Problem Statement:
# a) Create two 3x3 matrices A and B with numeric values.
# b) Compute A + B and A - B.
# c) Compute element-wise product of A and B.
# d) Compute matrix product of A and B using @ operator.
# e) Find transpose of A.
# f) Compute determinant and rank of A using numpy.linalg.
# g) Compute inverse of A if determinant != 0 and verify A @ inv(A) = I.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 7: Matrix Operations and Linear Algebra")
print("=" * 70)

# a) Define 3x3 matrices A and B
A7 = np.array([[2, 1, 3],
               [0, 5, 6],
               [7, 8, 9]], dtype=float)

B7 = np.array([[1, 0, 2],
               [3, 4, 5],
               [6, 7, 8]], dtype=float)

# b) Addition and Subtraction
print("\nb) A + B:\n", A7 + B7)
print("   A - B:\n", A7 - B7)

# c) Element-wise product
print("\nc) Element-wise Product (A * B):\n", A7 * B7)

# d) Matrix Multiplication
print("\nd) Matrix Product (A @ B):\n", A7 @ B7)

# e) Transpose
print("\ne) Transpose of A:\n", A7.T)

# f) Determinant and Rank
det_A7 = np.linalg.det(A7)
rank_A7 = np.linalg.matrix_rank(A7)
print(f"\nf) Determinant of A: {det_A7:.4f}")
print("   Rank of A:", rank_A7)

# g) Inverse & Identity Verification
if det_A7 != 0:
    A7_inv = np.linalg.inv(A7)
    print("\ng) Inverse of A:\n", A7_inv)
    print("   Verification (A @ A_inv ≈ Identity Matrix):\n", np.round(A7 @ A7_inv, 4))


# =============================================================================
# Question 8: Solving a Linear System
# =============================================================================
# Problem Statement:
# a) Create a 4x4 matrix A and a known solution vector x_true = [1, 2, 3, 4].
# b) Compute b = A @ x_true.
# c) Solve Ax = b using np.linalg.solve().
# d) Verify x matches x_true.
# e) Solve Ax = b by explicitly computing inv(A) @ b.
# f) Compare execution / numerical difference between the two approaches.
# g) Explain why np.linalg.solve() is preferred over explicit matrix inverse.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 8: Solving a Linear System")
print("=" * 70)

# a) Define coefficient matrix A and x_true
A8 = np.array([[3, 1, 2, 4],
               [1, 2, 1, 1],
               [2, 3, 2, 2],
               [1, 1, 1, 3]], dtype=float)
x_true = np.array([1, 2, 3, 4], dtype=float)

# b) Compute vector b
b8 = A8 @ x_true
print("\nb) Computed RHS vector b:", b8)

# c) Solve using np.linalg.solve()
x_solve = np.linalg.solve(A8, b8)
print("\nc) Solution via np.linalg.solve():", x_solve)

# d) Verify exact match
print("d) Solution matches x_true exactly:", np.allclose(x_solve, x_true))

# e) Solve via explicit matrix inversion: x = inv(A) @ b
x_inv = np.linalg.inv(A8) @ b8
print("\ne) Solution via np.linalg.inv(A) @ b:", x_inv)

# f) Absolute difference between both methods
print("f) Maximum Absolute Difference:", np.max(np.abs(x_solve - x_inv)))

# g) Explanation:
# np.linalg.solve() uses LU decomposition which is faster O(2/3 n^3) and numerically
# more stable against rounding errors compared to computing explicit matrix inverse.


# =============================================================================
# Question 9: Vectorized Polynomial and Vandermonde Matrix
# =============================================================================
# Problem Statement:
# a) Create vector x with values 1 through 10.
# b & c) Construct matrix P of shape (10, 4) with columns x, x^2, x^3, x^4.
# d) Construct 10x5 Vandermonde-style matrix with powers 0 to 4 using broadcasting.
# e) Verify first column contains all ones.
# f) Given coefficients c = [1, 2, 3, 4, 5], evaluate polynomial y = c0 + c1*x + ... + c4*x^4.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 9: Vectorized Polynomial and Vandermonde Matrix")
print("=" * 70)

# a) Vector x from 1 to 10
x9 = np.arange(1, 11)

# b & c) Matrix P (10, 4) with columns x, x^2, x^3, x^4
P9 = np.column_stack([x9**1, x9**2, x9**3, x9**4])
print("\nc) Matrix P (shape 10x4):\n", P9)

# d) Construct Vandermonde Matrix V with powers 0 to 4 using broadcasting
powers = np.arange(0, 5)
V = x9[:, None] ** powers
print("\nd) Vandermonde Matrix V (shape 10x5):\n", V)

# e) Verification of column 0
print("\ne) First column contains all ones:", np.all(V[:, 0] == 1))

# f) Evaluate polynomial y = 1 + 2x + 3x^2 + 4x^3 + 5x^4 using matrix multiplication
coeffs = np.array([1, 2, 3, 4, 5])
y_poly = V @ coeffs
print("\nf) Evaluated Polynomial values for x = 1..10:\n", y_poly)


# =============================================================================
# Question 10: Combined NumPy Data Analysis Task
# =============================================================================
# Problem Statement:
# a) Generate a random matrix of shape (20, 5) with integers between 10 and 100.
# b) Print shape, dimensions, min, max, overall mean.
# c) Compute mean of each column and identify feature with highest mean.
# d) Standardize each column using broadcasting.
# e) Select all rows where first feature (column 0) > 50.
# f) Replace outlier standardized values (> 2 or < -2) with 0.
# g) Compute 5x5 covariance matrix across features.
# h) Compute 5x5 correlation matrix.
# i) Find 2D matrix indices of global min and max values.
# j) Summarize how axis operations, broadcasting, and boolean indexing worked together.
# =============================================================================

print("\n" + "=" * 70)
print("QUESTION 10: Combined NumPy Data Analysis Task")
print("=" * 70)

# a) Generate random dataset (20 observations, 5 features)
data10 = np.random.randint(10, 101, size=(20, 5))

# b) Dataset Metadata
print("\nb) Dataset Overview:")
print("   Shape:", data10.shape)
print("   Dimensions:", data10.ndim)
print("   Min Value:", data10.min())
print("   Max Value:", data10.max())
print("   Overall Mean:", np.round(data10.mean(), 2))

# c) Column Means & Max Column
col_means10 = data10.mean(axis=0)
max_mean_col = np.argmax(col_means10)
print("\nc) Column Means:", np.round(col_means10, 2))
print(f"   Feature with highest mean: Column Index {max_mean_col}")

# d) Standardize using broadcasting
col_stds10 = data10.std(axis=0)
standardized10 = (data10 - col_means10) / col_stds10

# e) Filter rows where feature 0 > 50
filtered_rows = data10[data10[:, 0] > 50]
print(f"\ne) Rows where Column 0 > 50 (Count: {len(filtered_rows)}):\n", filtered_rows)

# f) Replace outlier values (> 2 or < -2) with 0
clipped_data = standardized10.copy()
clipped_data[(clipped_data > 2) | (clipped_data < -2)] = 0
print("\nf) Outlier zeroing complete. Zeroed values count:", np.sum(clipped_data == 0))

# g) Covariance Matrix
cov_mat = np.cov(data10, rowvar=False)
print("\ng) Feature Covariance Matrix (5x5):\n", np.round(cov_mat, 2))

# h) Correlation Matrix
corr_mat = np.corrcoef(data10, rowvar=False)
print("\nh) Feature Correlation Matrix (5x5):\n", np.round(corr_mat, 2))

# i) Global Minimum and Maximum indices
min_pos = np.unravel_index(np.argmin(data10), data10.shape)
max_pos = np.unravel_index(np.argmax(data10), data10.shape)
print(f"\ni) Global Minimum at Position {min_pos} with Value = {data10[min_pos]}")
print(f"   Global Maximum at Position {max_pos} with Value = {data10[max_pos]}")

# j) Conceptual Summary:
# Axis operations calculated statistics across observations (axis=0),
# broadcasting eliminated explicit Python loops during scaling, and 
# boolean indexing provided fast, memory-efficient vector filtering and cleaning.

print("\n" + "=" * 70)
print("ALL 10 NumPy ASSIGNMENT QUESTIONS EXECUTED SUCCESSFULLY!")
print("=" * 70)
