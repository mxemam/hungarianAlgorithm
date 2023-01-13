//----------------------------------------------------------------------------------//
// MIT License
//
// Copyright (c) [2020-] [Mostafa Emam]
//
// Author(s): Mostafa Emam (mostafa.emam92@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//----------------------------------------------------------------------------------//

#include <iostream>
#include "HungarianAlgorithm.h"

bool test3x3Matrix();
bool test4x4Matrix(HungarianAlgorithm<float> &hungAlgProblem);
bool test5x4Matrix(HungarianAlgorithm<float> &hungAlgProblem);

int main(int argc, const char *argv[])
{
      Eigen::MatrixXd costFcnMatrix(4, 2);
  costFcnMatrix << 71.36, 32.97, 82.23, 84.51, 75.62, 70.86, 69.42, 87.11;
  // Initialize the Hungarian algorithm object with the costFcnMatrix
  auto problem = HungarianAlgorithm<double>(costFcnMatrix);
  // Solve the assignment problem 
  problem.SolveAssignmentProblem();
  // Get the assignment results
  Eigen::MatrixXi assignmentMatrix(4, 2);
  problem.GetAssignmentMatrix(assignmentMatrix);
  std::cout << assignmentMatrix;
  return 0;
    // Specify and run some simple tests
    std::vector<int> bTestsPassedVector = {false, false, false};
    // Test 3x3 <int> matrix
    bTestsPassedVector[0] = test3x3Matrix();
    // Create an object and use it to test 4x4 and 5x4 <float> matrices
    auto hungAlgProblem = HungarianAlgorithm<float>();
    bTestsPassedVector[1] = test4x4Matrix(hungAlgProblem);
    bTestsPassedVector[2] = test5x4Matrix(hungAlgProblem);

    // Add final info message
    if (bTestsPassedVector[0] && bTestsPassedVector[1] && bTestsPassedVector[2])
    {
        std::cout << "SUCCESS: All tests passed successfully!\n";
    }
    else
    {
        std::cout << "ERROR: An error occurred during the execution of one of the tests!\n";
    }
    return 0;
};

bool test3x3Matrix()
{
    bool testPassed = true;

    // Create and initialize the cost function matrix
    Eigen::Matrix3i costFcnMatrix;
    costFcnMatrix << 40, 60, 15,
        25, 30, 45,
        55, 30, 25;
    // Initialize the Hungarian algorithm object with the costFcnMatrix
    auto hungAlgProblem = HungarianAlgorithm<int>(costFcnMatrix);
    // Solve the assignment problem
    hungAlgProblem.SolveAssignmentProblem();

    // Get the assignment results
    Eigen::MatrixXi assignmentMatrix(3, 3);
    hungAlgProblem.GetAssignmentMatrix(assignmentMatrix);
    // Compare to the expected results
    Eigen::Matrix3i expectedMatrix;
    expectedMatrix << 0, 0, 1,
        1, 0, 0,
        0, 1, 0;
    if (assignmentMatrix == expectedMatrix)
    {
        std::cout << "Correct assignment for 3x3 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect assignment for 3x3 problem!\n";
    }

    // Also check the assignment indices as vectors
    std::vector<int> rowIndices(3), columnIndices(3);
    hungAlgProblem.GetAssignmentResults(rowIndices, columnIndices);
    // Compare to the expected results
    std::vector<int> checkRowIndices = {2, 0, 1};
    std::vector<int> checkColIndices = {1, 2, 0};
    if (rowIndices == checkRowIndices)
    {
        std::cout << "Correct row indexing for 3x3 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect row indexing for 3x3 problem!\n";
    }
    if (columnIndices == checkColIndices)
    {
        std::cout << "Correct col indexing for 3x3 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect col indexing for 3x3 problem!\n";
    }
    // Add trailing empty line
    std::cout << "\n";

    return testPassed;
}

bool test4x4Matrix(HungarianAlgorithm<float> &hungAlgProblem)
{
    bool testPassed = true;

    // Create and initialize the cost function matrix
    Eigen::Matrix4f costFcnMatrix;
    costFcnMatrix << 4.9, 2.6, 5.2, 7.8,
        8.1, 3.2, 10.1, 8.3,
        12.8, 5.3, 4.5, 5.1,
        6.2, 3.1, 7.9, 14.5;
    // Assign the cost function to the Hungarian algorithm object
    hungAlgProblem.SetCostFunctionMatrix(costFcnMatrix);
    // Solve the assignment problem
    hungAlgProblem.SolveAssignmentProblem();

    // Get the assignment results
    Eigen::MatrixXi assignmentMatrix(4, 4);
    hungAlgProblem.GetAssignmentMatrix(assignmentMatrix);
    // Compare to the expected results
    Eigen::Matrix4i expectedMatrix;
    expectedMatrix << 0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
        1, 0, 0, 0;
    if (assignmentMatrix == expectedMatrix)
    {
        std::cout << "Correct assignment for 4x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect assignment for 4x4 problem!\n";
    }

    // Also check the assignment indices as vectors
    std::vector<int> rowIndices(4), columnIndices(4);
    hungAlgProblem.GetAssignmentResults(rowIndices, columnIndices);
    // Compare to the expected results
    std::vector<int> checkRowIndices = {2, 1, 3, 0};
    std::vector<int> checkColIndices = {3, 1, 0, 2};
    if (rowIndices == checkRowIndices)
    {
        std::cout << "Correct row indexing for 4x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect row indexing for 4x4 problem!\n";
    }
    if (columnIndices == checkColIndices)
    {
        std::cout << "Correct col indexing for 4x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect col indexing for 4x4 problem!\n";
    }
    // Add trailing empty line
    std::cout << "\n";

    return testPassed;
}

bool test5x4Matrix(HungarianAlgorithm<float> &hungAlgProblem)
{
    bool testPassed = true;

    // Create and initialize the cost function matrix
    Eigen::MatrixXf costFcnMatrix(5, 4);
    costFcnMatrix << 18, 11, 16.9, 22,
        14, 19, 26, 18,
        21, 23, 35, 29,
        42, 27, 21, 17,
        16, 15, 28, 25;
    // Assign the cost function to the Hungarian algorithm object
    hungAlgProblem.SetCostFunctionMatrix(costFcnMatrix);
    // Solve the assignment problem
    hungAlgProblem.SolveAssignmentProblem();

    // Get the assignment results
    Eigen::MatrixXi assignmentMatrix(5, 4);
    hungAlgProblem.GetAssignmentMatrix(assignmentMatrix);
    // Compare to the expected results
    Eigen::Matrix<int, 5, 4> expectedMatrix;
    expectedMatrix << 0, 0, 1, 0,
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1,
        0, 1, 0, 0;
    if (assignmentMatrix == expectedMatrix)
    {
        std::cout << "Correct assignment for 5x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect assignment for 5x4 problem!\n";
    }

    // Also check the assignment indices as vectors
    std::vector<int> rowIndices(5), columnIndices(4);
    hungAlgProblem.GetAssignmentResults(rowIndices, columnIndices);
    // Compare to the expected results, -1 to indicate an unused (undefined) index
    std::vector<int> checkRowIndices = {2, 0, -1, 3, 1};
    std::vector<int> checkColIndices = {1, 4, 0, 3};
    if (rowIndices == checkRowIndices)
    {
        std::cout << "Correct row indexing for 5x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect row indexing for 5x4 problem!\n";
    }
    if (columnIndices == checkColIndices)
    {
        std::cout << "Correct col indexing for 5x4 problem\n";
    }
    else
    {
        testPassed = false;
        std::cout << "ERROR: Incorrect col indexing for 5x4 problem!\n";
    }
    // Add trailing empty line
    std::cout << "\n";

    return testPassed;
}