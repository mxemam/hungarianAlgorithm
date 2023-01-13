## Preface
This repo contains an implementation of the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) in c++ using the [Eigen 3.3.9](https://eigen.tuxfamily.org/) library for matrix manipulation. The implementation was done for educational purposes. License information can be found [here](#license).

The project uses CMAKE to simplify downloading the Eigen library and running some basic tests. Code examples can be found [here](#code-examples).

---
## Operation Method
The Hungarian algorithm is a combinatorial optimization algorithm that solves a given assignment problem in polynomial time. It was developed and published by Harold Kuhn in 1955 and later reviewed by James Munkres in 1957, thus it is also known as the Kuhn-Munkres algorithm[^1]. The algorithm determines the optimal one-to-one correspondence for a non-negative $n$ x $n$ cost function matrix, with the aim of minimizing the overall cost assignment.

The method of operation is explained as follows:

1. **[Step 1] Subtract row minima:** for each row in the cost matrix, the smallest element is to be subtracted from all elements in its corresponding row, which yields at least one zero in that row. This procedure is repeated for all rows until at least one zero exists per row.
2. **[Step 2] Subtract column minima:** similarly, the smallest element in each column is subtracted from all elements in its corresponding column to yield at least one zero per column in the cost matrix.
3. **[Step 3] Find the minimum number of lines to cover all zeroes in the cost matrix:** all zeroes in the cost matrix are to be covered by as few lines as possible, in which lines can cover either rows or columns. If the number of lines is less than the matrix size n, step 4 is executed in order to increase the number of zeroes or shift their position in the cost matrix. Otherwise, step 4 is skipped and the problem proceeds to the solution phase, i.e. step 5.
4. **[Step 4] Augment the cost matrix:** augmenting the cost matrix is performed by subtracting the smallest uncovered element from all of the uncovered elements, as well as adding it to any elements at the intersection between two lines. Steps 3-4 are repeated until the number of lines is no longer less than the matrix size, i.e., number of lines is equal to the matrix size n
5. **[Step 5] Find the optimal assignment:** finally, a matrix of possible candidates for assignment is reached based on the cost minimization criteria, in which the assignment resembles a bijection, i.e., each element of one set is paired with exactly one element of the other set.

The full sequence of operation is demonstrated in the following figure:

![Hungarian_Algorithm](https://user-images.githubusercontent.com/98018278/212209380-0925dec1-2777-448d-98c3-42ef6844315d.png)

In case a non-square $n$ x $m$ cost matrix exists, the matrix is expanded to form a $r$ x $r$ square matrix, where $r=max⁡(n,m)$ and the additional elements have the dummy value (∞) [^2]. Accordingly, the Hungarian algorithm can be used with no issues.

[^1]: Burkard R.; Dell'Amico M. and Martello S. (2009): Assignment Problems: Revised Reprint. Italy, ISBN 978-1-611-97222-1
[^2]: Zervos M. (2012): Real-Time Multi-Object Tracking Using Multiple Cameras. Ecole Polytechnique Fédérale de Lausanne. https://infoscience.epfl.ch/record/183295/files/Report.pdf, last visited on [11.11.2020]

---
## Code Examples

Testing a 3x3 matrix
```cpp
#include <iostream>
#include "HungarianAlgorithm.h"

int main()
{
  // Create and initialize the cost matrix
  Eigen::Matrix3f costFcnMatrix;
  costFcnMatrix << 10.5, 22, 18, 42, 5.9, 6, 71.2, 8.4, 69;
  // Initialize the Hungarian algorithm object with the costFcnMatrix
  auto problem = HungarianAlgorithm<float>(costFcnMatrix);
  // Solve the assignment problem 
  problem.SolveAssignmentProblem();
  // Get the assignment results
  Eigen::MatrixXi assignmentMatrix(3, 3);
  problem.GetAssignmentMatrix(assignmentMatrix);
  std::cout << assignmentMatrix; // [1, 0, 0; 0, 0, 1; 0, 1, 0]
  return 0;
}
``` 

Testing a 4x2 matrix
```cpp
#include <iostream>
#include "HungarianAlgorithm.h"

int main()
{
  // Create and initialize the cost matrix
  Eigen::MatrixXd costFcnMatrix(4, 2);
  costFcnMatrix << 71.36, 32.97, 82.23, 84.51, 75.62, 70.86, 69.42, 87.11;
  // Initialize the Hungarian algorithm object with the costFcnMatrix
  auto problem = HungarianAlgorithm<double>(costFcnMatrix);
  // Solve the assignment problem 
  problem.SolveAssignmentProblem();
  // Get the assignment results
  Eigen::MatrixXi assignmentMatrix(4, 2);
  problem.GetAssignmentMatrix(assignmentMatrix);
  std::cout << assignmentMatrix; // [0, 1; 0, 0; 0, 0; 1, 0]
  return 0;
}
``` 

---
## License
This repo is available under the [MIT License](https://choosealicense.com/licenses/mit).
