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

#ifndef HUNGARIANALGORITHM_H_
#define HUNGARIANALGORITHM_H_

#include <Eigen/Dense>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <map>

// Check if a value is approximately zero (only positive values are expected in the
// cost function). The macro is better here as it is used for simple values, arrays,
// and matrices.
#define IsApproxZERO(X) ((X) <= 1e-6)

//----------------------------------------------------------------------------------//
// Enumeration for the state of the assignment problem
//
//  ELEMENTS
//      NotReady:       Problem initialized without cost function
//      ReadyToSolve:   Problem initialized with cost function, ready to solve
//      Done:           Problem sovled
//----------------------------------------------------------------------------------//
enum ProblemStatus
{
    NotReady,
    ReadyToSolve,
    Done
};
static std::map<ProblemStatus, const char *> ProblemStatusName = {
    {NotReady, "NotReady"},
    {ReadyToSolve, "ReadyToSolve"},
    {Done, "Done"}};

//----------------------------------------------------------------------------------//
// An implementation of the Hungarian algorithm to solve optimal assignment problems.
// The supported cost function matrix types are <int>, <float>, and <double>, and the
// result assignment matrix has the type <int>.
//
// Example:
//      Eigen::Matrix3f costFcnMatrix;
//      costFcnMatrix << 10.5, 22, 18, 42, 5.9, 6, 71.2, 8.4, 69;
//      auto problem = HungarianAlgorithm<float>(costFcnMatrix);
//      problem.SolveAssignmentProblem();
//      Eigen::MatrixXi assignmentMatrix(3, 3);
//      problem.GetAssignmentMatrix(assignmentMatrix);
//      std::cout << assignmentMatrix; // [1, 0, 0; 0, 0, 1; 0, 1, 0]
//----------------------------------------------------------------------------------//
template <typename T>
class HungarianAlgorithm
{
private:
    // Dimensions of the cost function matrix
    int nrRows, nrCols;
    // Size of the cost function matrix
    int matrixSize;
    // Dummy cost to indicate a very large number (Inf)
    double dummyCost;
    // Original cost function matrix
    Eigen::Matrix<T, -1, -1> costFunctionMatrix;
    // Editable work matrix
    Eigen::Matrix<T, -1, -1> workingMatrix;
    // Matrix used to determine the checked/covered elements
    Eigen::Array<bool, -1, -1> coveredMatrix;
    // Minimum number of lines needed to cover all the zeroes in the workingMatrix
    int nrLinesToCoverZeroes;
    // Assignment matrix
    Eigen::Array<bool, -1, -1> assignmentMatrix;
    // Variable to indicate current status
    ProblemStatus problemStatus = ProblemStatus::NotReady;

    // Step 1: Subtract row minima
    void SubtractRowMinima();
    // Step 2: Subtract column minima
    void SubtractColMinima();
    // Step 3: Get the minimum number of lines to cover all the zeroes
    int MinNrOfLinesToCoverAllZeros();
    // Step 4: Augment the matrix (Create additional zeroes)
    void AugmentCostFunctionMatrix();
    // Step 5: Find optimal cost
    void FindOptimalCost();

public:
    // Default object constructor, cost function matrix must be set later
    explicit HungarianAlgorithm();
    // Create the Hungarian algorithm object using the costFunctionMatrix
    HungarianAlgorithm(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &costFcnMatrix);

    // Set the cost function matrix
    void SetCostFunctionMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &costFcnMatrix);
    // Get the cost function matrix
    void GetCostFunctionMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &outMatrix);
    // Get the assignment matrix after solving the problem
    void GetAssignmentMatrix(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &outMatrix);
    // Get the assignment indices in two vectors for easy access
    void GetAssignmentResults(std::vector<int> &idxRow, std::vector<int> &idxCol);
    // Get current problem status
    ProblemStatus getProblemStatus() { return problemStatus; };
    // Get current problem status name
    std::string getProblemStatusName() { return ProblemStatusName[problemStatus]; };

    // Wrapper to execute all steps of the Hungarian algorithm and solve the assignment problem
    void SolveAssignmentProblem();
};

#endif // HUNGARIANALGORITHM_H_