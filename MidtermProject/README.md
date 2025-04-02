ssrini27 Midterm Project running instructions:

1. Please use a node with either 4, 16, 64 ... cores (perfect square)

2. Running script

cd denseGEMM-SUMMA
make clean
make
make {testcase}

3. Here, the testcase can be one of the following:

runsmall1
runsmall2
runmedium1
runmedium2
runlarge
runrectsmall1
runrectsmall2
runrectmedium1
runrectmedium2
runrectlarge

4. The make option will run both SUMMA-A & C for the test case and print the results.

5. Below is a script to run all test cases:

echo Running vsmall
make clean
make
make runvsmall

echo Running Small 1
make clean
make
make runsmall1

echo Running Small 2
make clean
make
make runsmall2

echo Running medium 1
make clean
make
make runmedium1

echo Running medium 2
make clean
make
make runmedium2

echo Running large
make clean
make
make runlarge

echo running rectangular

echo Running small1
make clean
make
make runrectsmall1

echo Running small2
make clean
make
make runrectsmall2

echo Running medium1
make clean
make
make runrectmedium1

echo Running medium2
make clean
make
make runrectmedium2

echo Running large
make clean
make
make runrectlarge
