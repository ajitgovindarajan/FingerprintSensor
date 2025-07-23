# FingerprintSensor

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Description of the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. Generate 100,000 random integers between 0 - 1,000,000.

Then save them to a text file where each line has 5 numbers per line.

1. Read the numbers back into a plain old array of integers.
1. Use insertion sort to sort the array
1. Asks the user to enter a number between 0 - 1,000,000.

The program uses the binary search algorithm to determine

if the specified number is in the array or not.  It also

displays the search step in details

1. Maintain a loop asking if the user wants to play again or not

after a search successfully completes.  Thetest set includes

the following integer numbers.

{-100, 0, 123456, 777777, 459845, 1000000, 1000001}


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Source files

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Name:  main.cpp

Main program.  This is the driver program that calls sub-functions

to read data from an input file, use the data to create two matrices,

and compute and display their sum and product.

Name:  xxx.h (if you have one)

Contains the definition for the class xxx.

Name: xxx.cpp

Defines and implements the xxx class for implementing a xxx.

This class provides routines to construct and get the xxx, ...



`	`[More text comes here ....]

Name: yyy.h

Contains the prototypes for the xxx program support functions.

Name: yyy.cpp

Includes functions to display a greeting, populate two arrays

from a data file, and display the sum and product of two matrices.


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Circumstances of programs

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

The program runs successfully.

The program was developed and tested on gnu g++ 6.1.x  It was

compiled, run, and tested on gcc csegrid server.


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* How to build and run the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. Uncompress the homework.  The homework file is compressed.

To uncompress it use the following commands

% unzip [1234HW1]

Now you should see a directory named homework with the files:

main.cpp

xxx.h

xxx.cpp

`	`yyy.h

`	`yyy.cpp

makefile

Readme.txt

`	`[any other supporting documents]

1. Build the program.

Change to the directory that contains the file by:

% cd [1234HW1]

Compile the program by:

% make

1. Run the program by:

% ./[xxx]

1. Delete the obj files, executables, and core dump by

%./make clean
