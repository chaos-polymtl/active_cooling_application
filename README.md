# Active cooling application

This software was developed to control the Active Cooling Device. The device is a flexible experimental unit, used for laboratory scale tests of the concept design developed and patented by [Lamarre & Raymond (2022)](https://patents.google.com/patent/US20230182361A1/en?oq=US20230182361A1).

## Instalation procedure

Installing the software is a straighforward process. After cloning the repository, open a terminal and navigate to the folder where the source code is located and follow the steps:

### Dependencies

The dependencies can be installed launching the following:

<pre> pip install -r requirements.txt </pre>

or

<pre> pip3 install -r requirements.txt </pre>


### Active cooling application

The software can be installed running:

<pre> pip install . </pre>

or

<pre> pip3 install . </pre>


## Testing the application

The application has a test mode, which is suitable for other Linux systems. This mode emulates the behavior of the application, using a dummy dataset instead of measuring quantities.

If you wish to verify your installation or run the application on test mode, launch the following command:

<pre> active-cooling-test $NUMBER_OF_REGIONS </pre>

## Launching the application

To launch the application, you can simply run:

<pre> active-cooling $NUMBER_OF_REGIONS </pre>

Where $NUMBER_OF_REGIONS should reflect the number of Mass Flow Controllers (MFCs)/solenoid valves connected to your system.

This application is meant to run on a Raspberry Pi. As such, it depends on the RPi module. As such, only the test mode would work on another system. Additionally, the I2C connection is tested upon launching the application. Hence, all hardware components should be connected upon application launching.

