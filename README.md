# FingerprintSensor

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Description of the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. This file will mimic the finger printsensor capabilities that is on smart phones
2. The program will consist of user interface that interact with the physical device with enterpirse integration, which contains the full facet of the application layer.
3. The next part is of the development side which will contain the neural network that will learn the patterns of the finger printand then store theright fingerprint as well as clean any other leanring that has come out of the attempts.
4. The we have the production core which implements the best of realtime processing, hardware capabilities and the core algorithms need for cyber physical systems use. 


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Source files

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Name:  matching.cpp

This file represents the fingerprint matching capabilites of the program along with the core algorithms for further implementation. Here we also have the hardware driver capability for cyber physical systems.

Name:  UI Interface.java

This file contains the user interface  with an application layer to all bring it together we have the enterprise integration. This operates as a sort of middle man for the sensor capability.

Name: Version2.py

This file has the model development, the preprocessing of the data as well as the research done through graph neural networks to learn as well as curb the storage capability of the learning process.

Name: Version2.py

This file has the model development, the preprocessing of the data as well as the research done throigh the traditional approach to neural netwroks like PyTorch through one true neural network


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* How to build and run the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. 
Now you should see a directory named homework with the files:

matching.cpp

UI interface.java

Version1.py

version2.py

makefile

Readme.txt



1. Build the program.

Change to the directory that contains the file by:

% cd [1234HW1]

Compile the program by:

% make

1. Run the program by:

% ./[xxx]

1. Delete the obj files, executables, and core dump by

%./make clean
