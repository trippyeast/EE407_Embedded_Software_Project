#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 
#include <sstream>
#include <stdio.h>
#include <opencv2/cudacodec.hpp>
#include <math.h>
#include <vector>
#include <utility>

	#include <fcntl.h>  /* File Control Definitions          */
	#include <termios.h>/* POSIX Terminal Control Definitions*/
	#include <unistd.h> /* UNIX Standard Definitions         */
	#include <errno.h>  /* ERROR Number Definitions          */

// Select Video Source
//#define TEST_LIVE_VIDEO

// AVG # of position values for Average
#define AVG 32

// PRECISION controls how far the line is drawn each interation of SearchForMovement()
// Larger values will increment the expected path by a smaller amount
// Smaller values are less accurate since an increment can overshoot the boundary
#define PRECISION 8

// Max # of reflections to draw 
#define MAX_REFLECTIONS 2

// Region of Interest Boundaries 
// (X1,Y1) is the top left corner
#define X1_BOUNDARY 520
#define X2_BOUNDARY 824
#define Y1_BOUNDARY 92
#define Y2_BOUNDARY 620

// Region of Interest of display region (window size)
#define X1_WINDOW 400
#define X2_WINDOW 1000
#define Y1_WINDOW 10
#define Y2_WINDOW 700

// Block and Aperture size for finding corners
#define blksz 2  
#define aperture 7 


using namespace cv;
using namespace std;

// Debugging Output Filestream 
std::ofstream fDebug("lab6_code_debug.txt");


int theObject[2] = {0,0};
int theObject_prev[2] = {0,0};

// Search For Movement
// Global variables and flags
int dex = 0;
int i = 0; int i_m1 = 0; int i_m2 = 0;
int x_array[AVG], y_array[AVG], zero_array[AVG];
double slope_array[AVG]; double dx_array[AVG]; double dy_array[AVG];
int *expected; // Pointer to hold expected position values			
int full_flag = 0;
int sign_change_flag = 0; 
int dx0_flag = 0;
int break_flag = 0;
bool firstFrame = true; 
bool badData = false;

// bounding rectangle of the object, we will use the center of this as its position.
cv::Rect objectBoundingRectangle = cv::Rect(0,0,0,0);

// Color Filtering Variables
// these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
// default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;
// minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
// names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

// Color Filter Presets
int filterTop[6] = {36, 256, 96, 256, 208, 256};
int filterBot[6] = {27, 52, 28, 80, 227, 256};
int filterPlayfield[6] = {0, 256, 0, 256, 110, 256};

// ROI for paddles
cv::Rect topRegion(518, 10, 310, 80);
cv::Rect botRegion(518, 623, 310, 82);

// Corners for warp transform
vector<Point2f> corners;
int Corner_Cnt = 0;

//cv::Rect botRegion(X1_BOUNDARY, Y2_BOUNDARY, X2_BOUNDARY-X1_BOUNDARY, regionSize);

//-----------------------------------------------------------------------------------------------------------------
// Functions
//-----------------------------------------------------------------------------------------------------------------
void on_trackbar( int, void* )
{//This function gets called whenever a
	// trackbar position is changed
}

//-----------------------------------------------------------------------------------------------------------------
// float to string helper function
//-----------------------------------------------------------------------------------------------------------------
string floatToString(float number){
 
    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}

//-----------------------------------------------------------------------------------------------------------------
// Int to String
//-----------------------------------------------------------------------------------------------------------------
string intToString(int number){
	std::stringstream ss;
	ss << number;
	return ss.str();
}

//-----------------------------------------------------------------------------------------------------------------
// Print an array of INT of size length
//-----------------------------------------------------------------------------------------------------------------
void printArray(int the_array[], int length){
	// cout << "size= " << sizeof(the_array) << endl;
	// memcpy(the_array, test_array, 8);
	for(int n=0; n<length ;n++)
	{
		cout << the_array[n] << ",";
	}
	cout << endl;
}

//-----------------------------------------------------------------------------------------------------------------
// Calculate the running average array value
//-----------------------------------------------------------------------------------------------------------------
// dx_avg = (slope_array[0] + slope_array[1] + ...) / index+1
double averageArray(int _dex, int _i, double *array) {
double sum = 0;
int a = 0;

	if (_dex == 1)
	a = 1;
	else if(_dex > AVG-1)
	a = AVG;
	else
	a = _i;

	for(int n = 0; n < a ; n++)
	sum = sum + array[n];
	
//fDebug << "AVG = sum/i = " << sum << " / " << a << endl;
//fDebug << "= " << sum/a << endl;

return sum/a;

}

//-----------------------------------------------------------------------------------------------------------------
// Create Trackbar Window
//-----------------------------------------------------------------------------------------------------------------
void createTrackbars(){
	//create window for trackbars

    namedWindow(trackbarWindowName,0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf( TrackbarName, "H_MIN", H_MIN);
	sprintf( TrackbarName, "H_MAX", H_MAX);
	sprintf( TrackbarName, "S_MIN", S_MIN);
	sprintf( TrackbarName, "S_MAX", S_MAX);
	sprintf( TrackbarName, "V_MIN", V_MIN);
	sprintf( TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );


}


//-----------------------------------------------------------------------------------------------------------------
// Print an array of Float of size length
//-----------------------------------------------------------------------------------------------------------------	
void printFloatArray(double the_array[], int length){
	// fDebug << "size= " << sizeof(the_array) << endl;
	// memcpy(the_array, test_array, 8);
	for(int n=0; n<length ;n++)
	{
		fDebug << the_array[n] << ",";
	}
	fDebug << endl;
}


//-----------------------------------------------------------------------------------------------------------------
// Calculate Expected Position Function
// 
// Implemented Recursively.
// The initial implementation using slope and similar triangles was too hard to troubleshoot
//-----------------------------------------------------------------------------------------------------------------
int* expectedY (int x_current, int y_current, int _direction, double _dx, double _dy) {
fDebug << "Calculating Expected (Y) Position from (" << x_current << "," << y_current << ") with Direction = " << _direction << " and differentials [" << _dx << " , " << _dy << "]" << endl;

static int nextPosition[2];
int nudge = 10;
int timeout = 0;
double y_e = y_current; 
double x_e = x_current;

	// Precision controls how far the line is drawn each interation
	

	// Non-zero Slope Directions
	if(_direction == 1 || _direction == 2 || _direction == 3 || _direction == 4) {

		_dy = _dy / PRECISION;
		_dx = _dx / PRECISION; 

		while (timeout < 100*PRECISION && x_e > X1_BOUNDARY-nudge && x_e < X2_BOUNDARY+nudge ) {
			y_e = y_e + _dy*timeout;
			x_e = x_e + _dx*timeout;
			timeout++;
			fDebug << x_e << ",";
		}
		fDebug << endl;
		fDebug << "Finished Calculating Position after " << timeout << " iterations" << endl;
		if(timeout == 0)
		fDebug << "---> ERROR: iterations = 0 " << endl;

		fDebug << "---> X boundary = " << x_e;
		fDebug << "---> Y boundary = " << y_e; 
	}
		
	
	// Zero Slope Directions
	if ( _direction == 1 || _direction == 2) 
		x_e = X2_BOUNDARY;
	
	else if(_direction == 3 || _direction == 4)
		x_e = X1_BOUNDARY;
	
	else if(_direction == 5) {
		y_e = Y1_BOUNDARY;
		x_e = x_current;
	}

	else if (_direction == 6){
		y_e = Y2_BOUNDARY;
		x_e = x_current;
	}	
		
		nextPosition[0] = x_e;
		nextPosition[1] = y_e;

return nextPosition;

}

//-----------------------------------------------------------------------------------------------------------------
// Bound the Expected Position by a Y border based on its direction
// 
// Also initially using similar triangles, but changed to recursive implementation
// Called after expectedY if the expected Y Value is behind the paddles (i.e. no pending reflection)
//-----------------------------------------------------------------------------------------------------------------
int* expectedX (int x_current, int y_current, int _direction, double _dx, double _dy){
// Calculate expected X (bounded by Y_Boundary)
// Using Similar triangles ( tan(d) = a/c = A/C, c = aC/A ) 
fDebug << "Calculating Expected (X) Position from (" << x_current << "," << y_current << ") with Direction = " << _direction << " and differentials [" << _dx << " , " << _dy << "]" << endl;

	static int nextPosition[2];
	int nudge = 10;

	_dy = _dy / (double)PRECISION;
	_dx = _dx / (double)PRECISION;
	
	// Iterate Recursively on timeout
	int timeout = 0;
	double y_e = y_current; 
	double x_e = x_current;

	
	// Non-zero Slope Directions
	if(_direction == 1 || _direction == 2 || _direction == 3 || _direction == 4) {

		while (timeout < 100*PRECISION && y_e > Y1_BOUNDARY-nudge && y_e <= Y2_BOUNDARY+nudge ) {
			y_e = y_e + _dy*timeout;
			x_e = x_e + _dx*timeout;
			timeout++;
		}
	
		fDebug << "Finished Calculating Position after " << timeout << "iterations" << endl;
		if(timeout == 0)
		fDebug << "---> ERROR: interations = 0 " << endl;

		fDebug << "---> X boundary = " << x_e;
		fDebug << "---> Y boundary = " << y_e; 
	}

	
	// Zero Slope Directions
	if ( _direction == 1 || _direction == 4) 
	y_e = Y1_BOUNDARY;

	else if(_direction == 2 || _direction == 3)
	y_e = Y2_BOUNDARY;

	else if(_direction == 5) {
		y_e = Y1_BOUNDARY;
		x_e = x_current;
	}

	else if (_direction == 6){
		y_e = Y2_BOUNDARY;
		x_e = x_current;
	}	

nextPosition[0] = x_e;
nextPosition[1] = y_e;		
		
return nextPosition;

}

//-----------------------------------------------------------------------------------------------------------------
// Calculate Slope from Direction and Expected Y Intersection Value
//-----------------------------------------------------------------------------------------------------------------
// Use the slope to predict where the ball will land on the Y axis
// direction = 1,2,3,4,5,6
//    5
// 4  |  1
//----|---
// 3  |  2
//    6	
int findDirection(double _slope, int _dx, int _dy) {

	int _direction = 10;
 	fDebug << "slope, dx, dy = " << _slope << "," << _dx << "," << _dy << endl;

	if (abs(_dx) < 1) {
		if(_dy < 0)
			_direction = 5;
		else if (_dy > 0)
			_direction = 6;
		else
			_direction = 7;
	}

	else {
		if (_dy < 0 && _dx > 0) 			
			_direction = 1;
		else if (_dy > 0 && _dx > 0) 		
			_direction = 2;
		else if (_dy > 0 && _dx < 0) 			
			_direction = 3;
		else if (_dy < 0 && _dx < 0) 			
			_direction = 4;
	}
	
return _direction;

}		


//-----------------------------------------------------------------------------------------------------------------
// Search for Moving Object Function
//-----------------------------------------------------------------------------------------------------------------
void searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed){
    //notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
    //to take the values passed into the function and manipulate them, rather than just working with a copy.
    //eg. we draw to the cameraFeed to be displayed in the main() function.
    bool objectDetected = false;
    int xpos, ypos;
	double m_avg, sum, dy, dx, dx_avg, dy_avg;
	int direction = 0;
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::Mat temp;
    thresholdImage.copyTo(temp);

#ifdef TEST_LIVE_VIDEO

    //find contours of filtered image using openCV findContours function
    cv::findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

#else

	double x1 = X1_BOUNDARY;
	double y1 = Y1_BOUNDARY;
	double x2 = X2_BOUNDARY;
	double y2 = Y2_BOUNDARY;

	cv::Rect roi(x1, y1, x2-x1, y2-y1);
	
    cv::Mat roi_temp = temp(roi); 

    //find contours of filtered image using openCV findContours function
    cv::findContours(roi_temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

#endif

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)
	objectDetected = true;
    else 
	objectDetected = false;
 
    if(objectDetected){

        // the largest contour is found at the end of the contours vector
        // we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));

        // make a bounding rectangle around the largest contour then find its centroid
        // this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));
	
		// Previous Frames position data
		theObject_prev[0] = theObject[0];
		theObject_prev[1] = theObject[1]; 

		// Calculate new position data
        xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;
 
        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }

//-----------------------------------------------------------------------------------------------------------------
//-------------------------------------------MOTION PREDICTION-----------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------
		
		fDebug << endl;
		fDebug << endl;

		fDebug << "--Start Motion Prediction--" << endl;
		fDebug << "Current Indexing [dex, i, i-1] = ";
		// Print Index Info
		fDebug << "[ " << dex << " , " << i << " , " << i_m1 << " ]" << endl;
		

		if(badData) {
			fDebug << "   Bad Data Previously Detected" << endl;

			if(sign_change_flag == 1) {
				fDebug << "   Direction Change Detected" << endl;
				sign_change_flag = 0;				
				for(int n = 0; n < AVG; n++) // Reset arrays on sign change
				{ 
					x_array[n] = 0;
					y_array[n] = 0;
					slope_array[n] = 0;
					dx_array[n] = 0;
					dy_array[n] = 0;
				}
				dex = 0; // Reset the index on sign change
			}

			else if(break_flag == 1) {
				fDebug << "   Continuity Break Previously Detected" << endl;
				break_flag = 0;				
					for(int n = 0; n < AVG; n++) // Reset arrays on sign change
					{ 
						x_array[n] = 0;
						y_array[n] = 0;
						slope_array[n] = 0;
						dx_array[n] = 0;
						dy_array[n] = 0;
					}

				dex = 0; // Reset the index on continuity break
			}
			
				break_flag = 0;
				dx0_flag = 0;
				badData = false;

		}

//-----------------------------------------------------------------------------------------------------------------
// Array Indexing
//-----------------------------------------------------------------------------------------------------------------		
		else {

			i = dex % (int)AVG; // index mapped to 0-AVG-1 for calculations

			i_m1 = (dex-1) % (int)AVG; // previous index (accounting for looparound)
			if (i_m1<0)
			i_m1 = 0;
		
			i_m2 = (dex-2) % (int)AVG; // previous index (accounting for looparound)		
			if (i_m2<0)
			i_m2 = 0;
		
			if(i == i_m1)
			firstFrame = true;
		
		  // fDebug << "Incrementing index" << endl;			
			// dex++
		  //fDebug << "Updated Indexing: ";
		  // Print Index Info
		  fDebug << dex << "," << i << "," << i_m1 << endl;	

//-----------------------------------------------------------------------------------------------------------------
// Grab New Position Values, offset by x1 and y1
//-----------------------------------------------------------------------------------------------------------------		
    	x_array[i] = theObject[0]+x1;
    	y_array[i] = theObject[1]+y1; 
	fDebug << "New Position Values (x,y) =" << "(" << theObject[0]+x1 << "," << theObject[1]+y1<< ")" << endl;
//-----------------------------------------------------------------------------------------------------------------
// First Frame, No calculation
//-----------------------------------------------------------------------------------------------------------------
		// Calculate position differentials
		if (firstFrame) {
			fDebug << "First Frame (Not calculating dx or dy), Printing x,y array..." << endl; // Confirm it happens
			printArray(x_array, AVG);
			printArray(y_array, AVG);
			firstFrame = false;
			dex++; 
		}

//-----------------------------------------------------------------------------------------------------------------
// Proceed when the array has at least 2 values
//-----------------------------------------------------------------------------------------------------------------
		else if (dex > 0) {
			fDebug << "Position array has >2 Good values... Confirm w. x,y array..." << endl;
			
			//printArray(x_array, AVG);
			//printArray(y_array, AVG);	

			dx = x_array[i] - x_array[i_m1]; 
			dy = y_array[i] - y_array[i_m1];
			fDebug << "dy = " << y_array[i] << " - " << y_array[i_m1] << " = " << dy << endl;
			fDebug << "dx = " << x_array[i] << " - " << x_array[i_m1] << " = " << dx << endl;	
			
			dx_array[i_m1] = dx; 
			dy_array[i_m1] = dy;	

			// Infinite/Undetermined Slope
			if (abs(dx) < 0.01) {
				fDebug << "BAD DATA: Dx=0, infinite slope " << endl;
				dx0_flag = 1;
				//badData = true;
				dx = 0;
			}

			// Zero Slope
			if(abs(dy) < .00001) 
			slope_array[i_m1] = 0.00;

			if (abs(dx) > 20 || abs(dy) > 20) {	
				fDebug << "Continuity Break" << endl;
				break_flag = 1;
				badData = true;
			}

			// Calculate Slope if Good Data
			if (!badData) {
				fDebug << "Good Data, Calculating Slope..." << endl;
				slope_array[i_m1] = (float)dy/(float)(dx);		

				fDebug << "Slope = " << slope_array[i_m1] << endl;
				fDebug << "i=" << i_m1 << endl;
				fDebug << "Slope Array: ";
				//printFloatArray(slope_array, AVG);

			// Check for sign change
			if((slope_array[i_m1] < 0 && slope_array[i_m2] >= 0 ) || (slope_array[i_m1] >= 0 && slope_array[i_m2] < 0)) // True if there was a sign swap in the slope array
			{
				// There has been a sign change... We can't calculate anything
				fDebug << "Sign Change Detected..." << endl;
				sign_change_flag = 1;
				badData = true;			
			}
}
				
//-----------------------------------------------------------------------------------------------------------------
// Calculating Expected Values  
//-----------------------------------------------------------------------------------------------------------------
			
			// Check for Bad Data
			if(badData)
				{
					fDebug << "Bad Data detected" << endl;
					if(sign_change_flag == 1)
					fDebug << "Sign Change" << endl;
				
					if(dx0_flag == 1)
					fDebug << "Infinite Slope" << endl;
		
					if(break_flag == 1)
					fDebug << "Continuity Break" << endl;
				}

			else // GOOD DATA: CALCULATE EXPECTED POSITION
			{		
				fDebug << "Incrementing index" << endl;			
				dex++;
				fDebug << "dx_array: ";
				printFloatArray(dx_array, AVG);
				fDebug << "dy_array: ";
				printFloatArray(dy_array, AVG);

				m_avg = averageArray(dex, i, slope_array);
				dx_avg = averageArray(dex, i, dx_array);
				dy_avg = averageArray(dex, i, dy_array);
	
				direction = findDirection(m_avg, dx, dy);
				
				expected = expectedY(x_array[i], y_array[i], direction, dx_avg, dy_avg);

				// Cut off the vector in the y-directions and calculate the expected x location
				int nudge = 10; // Draw the flight path past the ROI by nudge
				if (expected[1] <= (Y1_BOUNDARY - nudge) || expected[1] >= (Y2_BOUNDARY + nudge)) 
				expected = expectedX(x_array[i], y_array[i], direction, dx_avg, dy_avg);
				
				
				// Draw the first expected path
				line(cameraFeed,Point(x_array[i],y_array[i]),Point(expected[0],expected[1]),Scalar(255,0,0),2);

				fDebug << "Drawing from point (" << x_array[i] << "," << y_array[i] << ") To point (" << expected[0] << "," << expected[1] << ")" << endl;

				if(expected[1] - y_array[i] < dy || expected[0] - x_array[i] > dx) {
					fDebug << "ERROR - BAD EXPECTED POSITION... dx,dy = " << expected[0] - x_array[i] << "," << expected[1] - y_array[i] << endl;
				}			
//-----------------------------------------------------------------------------------------------------------------
// Reflections 
//-----------------------------------------------------------------------------------------------------------------	
				// Subsequent Vectors
				// Draw more vectors when the new y_expected value is within ROI
		if(direction == 1 || direction == 2 || direction == 3 || direction == 4) {
				int reflections = 0;
				int y_prev[8]; // Store y_expected value
				int x_prev[8]; // Store x_expected value	

								
				x_prev[0] = expected[0];
				y_prev[0] = expected[1];
				
				fDebug << "Current x_next, y_next = " << endl << expected[0] << "," << expected[1] << endl;				
	
					while (y_prev[reflections] > Y1_BOUNDARY-nudge && y_prev[reflections] < Y2_BOUNDARY+nudge && reflections < MAX_REFLECTIONS) {
						

						fDebug << "x_prev[]: ";
						printArray(x_prev, 8);
						fDebug << "y_prev[]: ";
						printArray(y_prev, 8);

						fDebug << endl << "Drawing Reflection #" << reflections+1 << endl;
						//m_avg = m_avg * (-1);
						
						fDebug << "Reflected Slope = " << m_avg << endl;
						fDebug << "Previous Direction, New Direction = " << direction;

							switch(direction) {
								case 1 :
									direction = 4;
									dx_avg = dx_avg * (-1);
									break;
								case 2 :
									direction = 3;
									dx_avg = dx_avg * (-1);
									break;
								case 3 :
									direction = 2;
									dx_avg = dx_avg * (-1);
									break;
								case 4 :
									direction = 1;
									dx_avg = dx_avg * (-1);
									break;
								case 5 : 
									direction = 6;
									dy_avg = dy_avg * (-1);
									break;
								case 6 :
									direction = 5;
									dy_avg = dy_avg * (-1);
									break;
								case 7 :
									direction =  7;
									break;									 
							}
						fDebug << " , " << direction << endl;
				
						expected = expectedY(x_prev[reflections], y_prev[reflections], direction, dx_avg, dy_avg);
						fDebug << "Current x_next, y_next = " << endl << expected[0] << "," << expected[1] << endl;
			
						// Cut off the vector in the y-directions and calculate the expected x location
						if (expected[1] <= (Y1_BOUNDARY - nudge) || expected[1] >= (Y2_BOUNDARY + nudge)) 
						expected = expectedX(x_prev[reflections], y_prev[reflections], direction, dx_avg, dy_avg);

						fDebug << "Bounded Current x_next, y_next = " << endl << expected[0] << "," << expected[1] << endl;
						
						reflections++;						
						y_prev[reflections] = expected[1];
						x_prev[reflections] = expected[0];
			
						
					}							
								
				fDebug << "Found " << reflections << " TOTAL Reflections" << endl;

				fDebug << "x_prev[]: ";
				printArray(x_prev, 8);
				fDebug << "y_prev[]: ";
				printArray(y_prev, 8);
				// Draw the new reflected vector(s)
				if (reflections == 0) 
					fDebug << "0 Reflections?" << endl;

				for(int l = 0; l<reflections; l++) {
					fDebug << "Drawing reflection #" << l << endl;
					line(cameraFeed,Point(x_prev[l],y_prev[l]),Point(x_prev[l+1],y_prev[l+1]),Scalar(255,0,0),2);
					fDebug << "Drawing from point (" << x_prev[l] << "," << y_prev[l] << ") To point (" << x_prev[l+1] << "," << y_prev[l+1] << ")" << endl;			
				}	
		 }
		
	}
}

} // end if(!badData)
	int x = x_array[i];
	int y = y_array[i];			
    // Draw some crosshairs around the object
    circle(cameraFeed,Point(x,y),10,cv::Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

	// Draw the ROI
    line(cameraFeed,Point(roi.x,roi.y),Point(roi.x+roi.width,roi.y),Scalar(255,0,0),2);
	line(cameraFeed,Point(roi.x,roi.y),Point(roi.x,roi.y+roi.height),Scalar(255,0,0),2);
	line(cameraFeed,Point(roi.x+roi.width,roi.y),Point(roi.x+roi.width,roi.y+roi.height),Scalar(255,0,0),2);
	line(cameraFeed,Point(roi.x,roi.y+roi.height),Point(roi.x+roi.width,roi.y+roi.height),Scalar(255,0,0),2);

    //write the position of the object to the screen
    putText(cameraFeed,"(" + intToString(x)+","+intToString(y)+","+floatToString(m_avg)+")",Point(x,y),1,1,Scalar(255,0,0),2);
	putText(cameraFeed,"slope ="+floatToString(m_avg),Point(20,20*3),1,1,Scalar(0,0,255),2);
	
	// Draw a short vector indicating the direction we think we are going
	/*
	int N = 25;
	if (direction == 1) {
		putText(cameraFeed, "UP to the RIGHT", Point(20, 20*3),1,1,Scalar(255,255,255),2);
		line(cameraFeed,Point(x,y),Point(x+N,y-N),Scalar(255,0,0),2); // Blue when going up
	}	

	else if (direction == 2) {
		putText(cameraFeed, "DOWN to the RIGHT", Point(20, 20*3),1,1,Scalar(255,255,255),2);
		line(cameraFeed,Point(x,y),Point(x+N,y+N),Scalar(0,0,255),2); // Red when going down			
	}

	else if (direction == 3) {
		putText(cameraFeed, "DOWN to the LEFT", Point(20, 20*3),1,1,Scalar(255,255,255),2);
		line(cameraFeed,Point(x,y),Point(x-N,y+N),Scalar(255,0,0),2); // Blue when going up
	}	

	else if (direction == 4) {
		putText(cameraFeed, "UP to the LEFT", Point(20, 20*3),1,1,Scalar(255,255,255),2);
		line(cameraFeed,Point(x,y),Point(x-N,y-N),Scalar(0,0,255),2); // Red when going down			
	}
 	*/


	
    // Print the object position
    // fDebug << xpos << " " << ypos << endl;
	fDebug << "END SEARCH FOR MOVEMENT" << endl;
} // searchForMovement




//-----------------------------------------------------------------------------------------------------------------
// Find Paddles
//-----------------------------------------------------------------------------------------------------------------
void locatePaddle(cv::Rect roi, cv::Mat thresholdImage, cv::Mat &cameraFeed) {

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	cv::Rect Pad = cv::Rect(0,0,0,0);
	cv::Mat temp;

	thresholdImage.copyTo(temp);
	cv::Mat roi_temp = temp(roi);

	cv::findContours(roi_temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );
	
	if(contours.size()>0) {
		vector< vector<Point> > largestContourVec;
		largestContourVec.push_back(contours.at(contours.size()-1));
		Pad = boundingRect(largestContourVec.at(0));
	}

	// Draw the ROI
	line(cameraFeed,Point(roi.x,roi.y),Point(roi.x+roi.width,roi.y), Scalar(255,0,255),2);
	line(cameraFeed,Point(roi.x,roi.y),Point(roi.x,roi.y+roi.height),Scalar(255,0,255),2);
	line(cameraFeed,Point(roi.x+roi.width,roi.y),Point(roi.x+roi.width,roi.y+roi.height),Scalar(255,0,255),2);
	line(cameraFeed,Point(roi.x,roi.y+roi.height),Point(roi.x+roi.width,roi.y+roi.height),Scalar(255,0,255),2);		

	Pad.x = Pad.x + roi.x;
	Pad.y = Pad.y + roi.y;

	// Draw the Paddle
	if(contours.size()>0) {
	line(cameraFeed,Point(Pad.x, Pad.y),Point(Pad.x+Pad.width, Pad.y),Scalar(0,200,0),2);
	line(cameraFeed,Point(Pad.x, Pad.y),Point(Pad.x, Pad.y+Pad.height),Scalar(0,200,0),2);	
	line(cameraFeed,Point(Pad.x+Pad.width, Pad.y),Point(Pad.x+Pad.width, Pad.y+Pad.height),Scalar(0,200,0),2);
	line(cameraFeed,Point(Pad.x+Pad.width, Pad.y+Pad.height),Point(Pad.x+Pad.width, Pad.y+Pad.height),Scalar(0,200,0),2);
	}	
}


//-----------------------------------------------------------------------------------------------------------------
// Find Corners
//-----------------------------------------------------------------------------------------------------------------
int * locateCorners(cv::Mat src, cv::Mat &cameraFeed, int &cornersFound){

	cout << endl << "START locateCorners" << endl;
    // Allocating size of image    
    cv::Mat dst(src.size(), CV_32FC1, Scalar::all(0));
    cv::Mat dst_norm(src.size(), CV_32FC1, Scalar::all(0));

    int thresh = 200;  //180
    int apertureSize = 9;
    int blockSize = 2;
    double harrisK = 0.04;

	double Give = 20;
	int t = 0;

	int theCorner[16]; // Store up to 16 corner point values
	static int finalCorner[8]; // [x1, y1, x2, y2...]

    // Detecting corners
    cv::cornerHarris( src, dst, blockSize, apertureSize, harrisK, BORDER_DEFAULT );

    // Normalize
    cv::normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );

  
    // Drawing circles around corners
    for( int j=0; j<dst_norm.rows; j++)
    {
        for( int i=0; i< dst_norm.cols; i++)
        {
            if( (int) dst_norm.at<float>(j,i) > thresh )
            {             
                cv::circle(cameraFeed, Point(i,j), 5, Scalar(55,255,55), 2, 8, 0);
                std::cout << "New Corner Point Found: (" << i << "," << j << ")" << endl;
						
				cout << "Corner Index (t) = " << t << endl;		

				if(t == 0) { 
					cout << "First Corner" << endl;				
					theCorner[t] = i;
					theCorner[t+1] = j;
					t = t + 2;
					printArray(theCorner, 16);
				}
		
				// The Array has positons to be filled
				else if (t >= 2 && t <=6) { 

					// If the new corner value is close to the previous corner then it is not saved
					// Tune with parameter Give
					if (theCorner[t-2] - Give < i && i < theCorner[t-2] + Give 	// New corner x-value is close to previous corner 
						&& 														// &&
						theCorner[t-1] - Give < j && j < theCorner[t-1] + Give) // New corner y-value is close to previous corner
					{
						cout << "New Corner " << "(" << i << "," << j << ")" 
							 << " is near the previous corner " << "(" << theCorner[t-2] << "," << theCorner[t-1] << ")" << endl;
		
						cout << theCorner[t-2] - Give << " < " << i << " < " << theCorner[t-2] + Give << endl;
						cout << theCorner[t-1] - Give << " < " << j << " < " <<  theCorner[t-1] + Give << endl;
					}

					// If the new corner value is far from previous corner value then save it and increment the index
					else 
					{
						cout << "Saving new corner" << endl;
						// Store X and Y corner value
						theCorner[t] = i;
						theCorner[t+1] = j;
						t = t + 2;
					}
				}
				
				// The Array is Full
				else
				{
					// This is an overflow in case more than 4 unique corners are detected
					cout << "Corner Array is Full" << endl;
					printArray(theCorner, 16);
				}
	
            }

        }
    }

		// Match the correct corner values
		cout << endl;
		cout << "Gathered " << t/2 << " corner values" << endl;
		printArray(theCorner, t);
		
		cornersFound = t/2;

		int cx; int cy;
		int X_MIDDLE = 650;
		int Y_MIDDLE = 500;

		// Match the Corners
		for(int it = 0; it < t; it += 2)
		{
			cx = theCorner[it];
			cy = theCorner[it+1];

			if (cx < X_MIDDLE && cy < Y_MIDDLE) {
				cout << "(" << cx << "," << cy << ") is the TOP LEFT CORNER" << endl;
				finalCorner[0] = cx;
				finalCorner[1] = cy;
			}  

			else if (cx > X_MIDDLE && cy < Y_MIDDLE) {
				cout << "(" << cx << "," << cy << ") is the TOP RIGHT CORNER" << endl;
				finalCorner[2] = cx;
				finalCorner[3] = cy;
			}  
		
			else if (cx > X_MIDDLE && cy > Y_MIDDLE) {
				cout << "(" << cx << "," << cy << ") is the BOTTOM RIGHT CORNER" << endl;
				finalCorner[4] = cx;
				finalCorner[5] = cy;
			}  

			else if (cx < X_MIDDLE && cy > Y_MIDDLE) {
				cout << "(" << cx << "," << cy << ") is the BOTTOM LEFT CORNER" << endl;
				finalCorner[6] = cx;
				finalCorner[7] = cy;
			}  
		}



cout << endl;
return finalCorner;

}

//-----------------------------------------------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------------------------------------------
int main() {

	memset (x_array, 0, AVG); //{1, 2, 3, 4, 5, 6, 7, 8};
    // OpenCV frame matrices
    cv::Mat frame0, frame1, result, frame0_warped, frame1_warped, threshold0, threshold1, cpu_HSV0, cpu_HSV1;
    cv::cuda::GpuMat gpu_frame0, gpu_frame1, gpu_grayImage0, gpu_grayImage1, gpu_differenceImage, gpu_thresholdImage, gpu_frame0_warped, gpu_frame1_warped, HSV0, HSV1;
    int toggle, frame_count;
	int cornerCount = 0;
	int *theCorners;
	int paddle[2] = {0, 0};
	
	char write_buffer[1];
  	char read_buffer[1];
  	int  bytes_written;  
  	int  bytes_read; 
  	struct termios options;           // Terminal options
  	int fd;                           // File descriptor for the port

	cv::Rect displayRegion(Point(X1_WINDOW,Y1_WINDOW), Point(X2_WINDOW, Y2_WINDOW));
	cv::Mat crop;
	cv::Mat filterThreshold;
	cv::Mat T;
	createTrackbars();

#ifdef TEST_LIVE_VIDEO
    // Camera video pipeline
    std::string pipeline = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

#else
    // MP4 file pipeline
    std::string pipeline = "filesrc location=/home/nvidia/Desktop/pong_video.mp4 ! qtdemux name=demux ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

#endif

    cout << "Using pipeline: " << pipeline << std::endl;
 
    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        cout << "Connection failed" << std::endl;
        return -1;
    }

//-----------------------------------------------------------------------------------------------------------------
// Arduino Interface
//-----------------------------------------------------------------------------------------------------------------
fd = open("/dev/ttyACM0",O_RDWR | O_NOCTTY);   // Open tty device for RD and WR
usleep(2000000); 
    if(fd == 1) {
        printf("\n  Error! in Opening ttyACM0\n");
    }
    else
        printf("\n  ttyACM0 Opened Successfully\n");

    tcgetattr(fd, &options);               // Get the current options for the port
    cfsetispeed(&options, B115200);        // Set the baud rates to 115200          
    cfsetospeed(&options, B115200);                   
    options.c_cflag |= (CLOCAL | CREAD);   // Enable the receiver and set local mode           
    options.c_cflag &= ~PARENB;            // No parity                 
    options.c_cflag &= ~CSTOPB;            // 1 stop bit                  
    options.c_cflag &= ~CSIZE;             // Mask data size         
    options.c_cflag |= CS8;                // 8 bits
    options.c_cflag &= ~CRTSCTS;           // Disable hardware flow control    

// Enable data to be processed as raw input
    options.c_lflag &= ~(ICANON | ECHO | ISIG);
     
    tcsetattr(fd, TCSANOW, &options);      // Apply options immediately
    fcntl(fd, F_SETFL, FNDELAY);    
    


//-----------------------------------------------------------------------------------------------------------------
// First Frame / Corner Detection
//-----------------------------------------------------------------------------------------------------------------
// Corner Detection

	do {
		cap >> frame0;
		
		// Convert to HSV		
		cvtColor(frame0, cpu_HSV0, COLOR_BGR2HSV);
						
		// Color Filtering
		//inRange(cpu_HSV0,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),filterThreshold);
		inRange(cpu_HSV0,Scalar(filterPlayfield[0], filterPlayfield[2], filterPlayfield[4]),Scalar(filterPlayfield[1], filterPlayfield[3], filterPlayfield[5]),filterThreshold);
		
		theCorners = locateCorners(filterThreshold, frame0, cornerCount);
		cout << "locateCorners found " << cornerCount << " corners" << endl;
	
	} while (cornerCount < 4);
	
	cout << "Using the following points for perspective transformation: " << endl;
	//printArray(theCorners,8); 
	cornerCount = 0;

	// Upload frame to GPU
	gpu_frame0.upload(frame0);
		

//-----------------------------------------------------------------------------------------------------------------
// Perspective Transform
//-----------------------------------------------------------------------------------------------------------------
		  
	// Input Quadilateral or Image plane coordinates
	Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	Point2f outputQuad[4];
	// Lambda Matrix (Transformation Matrix)
	cv::Mat lambda( 2, 4, CV_32FC1);

	// 4 points that select the quadilateral input from top left corner, then clockwise
	inputQuad[0] = Point(theCorners[0], theCorners[1]);
    inputQuad[1] = Point(theCorners[2], theCorners[3]);
    inputQuad[2] = Point(theCorners[4], theCorners[5]);
	inputQuad[3] = Point(theCorners[6], theCorners[7]);

	// The 4 points where the mapping is to be done, from top-left in clockwise order
	outputQuad[0] = Point2f(437, 0);
	outputQuad[1] = Point2f(842, 0);
	outputQuad[2] = Point2f(842, 719);
	outputQuad[3] = Point2f(437, 719);

	// Get the Perspective transfrom matrix (lambda)
	lambda = cv::getPerspectiveTransform(inputQuad,outputQuad);
 
	// Warp Perspective
	cv::cuda::warpPerspective(gpu_frame0,gpu_frame0_warped,lambda,gpu_frame0.size());

	// Convert to grayscale
    cv::cuda::cvtColor(gpu_frame0_warped,gpu_grayImage0,cv::COLOR_BGR2GRAY);

	// Download to CPU memory
	gpu_frame0_warped.download(frame0_warped);


//-----------------------------------------------------------------------------------------------------------------
// Toggle Loop
//-----------------------------------------------------------------------------------------------------------------
	// Initialize 
	toggle = 0;
	frame_count = 0;

    while (frame_count < 1000) {
		fDebug << "Frame #" << frame_count << endl;

		if (toggle == 0) {
            // Get a new frame from file
          
			if (frame_count %100 == 0) 
			{
				// Update the corner detection every 100 frames
				fDebug << "Recalculating Corners" << endl;
				do {
					cap >> frame1;
		
					// Convert to HSV		
					cvtColor(frame1, cpu_HSV1, COLOR_BGR2HSV);
						
					// Color Filtering
					//inRange(cpu_HSV0,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),filterThreshold);
					inRange(cpu_HSV1,Scalar(filterPlayfield[0], filterPlayfield[2], filterPlayfield[4]),
							Scalar(filterPlayfield[1], filterPlayfield[3], filterPlayfield[5]),filterThreshold);
		
					theCorners = locateCorners(filterThreshold, frame0, cornerCount);
					fDebug << "locateCorners found " << cornerCount << " corners" << endl;
	
				} while (cornerCount < 4);
	
				//cout << "Using the following points for perspective transformation: ";
				//printArray(theCorners,8); 
				cornerCount = 0;
			
				// Upload frame to GPU
				gpu_frame1.upload(frame1);

				// 4 points that select the quadilateral input from top left corner, then clockwise
				inputQuad[0] = Point(theCorners[0], theCorners[1]);
				inputQuad[1] = Point(theCorners[2], theCorners[3]);
				inputQuad[2] = Point(theCorners[4], theCorners[5]);
				inputQuad[3] = Point(theCorners[6], theCorners[7]);
			
				// Get the Perspective transfrom matrix (lambda)
				lambda = cv::getPerspectiveTransform(inputQuad,outputQuad);
			}

			else 
			{
				// Get a new frame from file
		        cap >> frame1;
			}

			// Upload to GPU memory
		   	gpu_frame1.upload(frame1);

			// Warp Perspective
			cv::cuda::warpPerspective(gpu_frame1,gpu_frame1_warped,lambda,gpu_frame1.size());

			// Convert to grayscale and HSV
			cv::cuda::cvtColor(gpu_frame1_warped,gpu_grayImage1,cv::COLOR_BGR2GRAY);
			cv::cuda::cvtColor(gpu_frame1_warped,HSV1,COLOR_BGR2HSV);

			// Download to cpu memory
			gpu_frame1_warped.download(frame1_warped);
			HSV1.download(cpu_HSV1);
			
			// Color Filtering for Top and Bottom paddles
			inRange(cpu_HSV1,Scalar(filterTop[0], filterTop[2], filterTop[4]),Scalar(filterTop[1], filterTop[3], filterTop[5]),threshold0);
			inRange(cpu_HSV1,Scalar(filterBot[0], filterBot[2], filterBot[4]),Scalar(filterBot[1], filterBot[3], filterBot[5]),threshold1);
	
			// Search for paddle locations
			locatePaddle(topRegion,threshold0,frame1_warped);
			locatePaddle(botRegion,threshold1,frame1_warped);

		    toggle = 1;
			
		}
		
		else //toggle == 0
		{ 
			if (frame_count %100 == 0) 
			{
				// Update the corner detection every 100 frames
				fDebug << "Recalculating Corners" << endl;
				do {
					cap >> frame0;
		
					// Convert to HSV		
					cvtColor(frame0, cpu_HSV0, COLOR_BGR2HSV);
						
					// Color Filtering
					//inRange(cpu_HSV0,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),filterThreshold);
					inRange(cpu_HSV0,Scalar(filterPlayfield[0], filterPlayfield[2], filterPlayfield[4]),
							Scalar(filterPlayfield[1], filterPlayfield[3], filterPlayfield[5]),filterThreshold);
		
					theCorners = locateCorners(filterThreshold, frame0, cornerCount);
					fDebug << "locateCorners found " << cornerCount << " corners" << endl;
	
				} while (cornerCount < 4);
	
				//fDebug << "Using the following points for perspective transformation: ";
				//printArray(theCorners,8); 
				cornerCount = 0;
			
				// Upload frame to GPU
				gpu_frame0.upload(frame0);

				// 4 points that select the quadilateral input from top left corner, then clockwise
				inputQuad[0] = Point(theCorners[0], theCorners[1]);
				inputQuad[1] = Point(theCorners[2], theCorners[3]);
				inputQuad[2] = Point(theCorners[4], theCorners[5]);
				inputQuad[3] = Point(theCorners[6], theCorners[7]);
			
				// Get the Perspective transfrom matrix (lambda)
				lambda = cv::getPerspectiveTransform(inputQuad,outputQuad);
			}

			else
			{
				// Get a new video frame	
		   	    cap >> frame0;
			}

			// Upload to GPU memory           
			gpu_frame0.upload(frame0);

			// Warp Perspective
			cv::cuda::warpPerspective(gpu_frame0,gpu_frame0_warped,lambda,gpu_frame0.size());
			
			// Convert to grayscale
			cv::cuda::cvtColor(gpu_frame0_warped,gpu_grayImage0,cv::COLOR_BGR2GRAY);
		    cv::cuda::cvtColor(gpu_frame0_warped,HSV0,COLOR_BGR2HSV);

			// Download to cpu memory
			gpu_frame0_warped.download(frame0_warped);
			HSV0.download(cpu_HSV0);

			// Color Filtering for Top and Bottom paddles
			inRange(cpu_HSV0,Scalar(filterTop[0], filterTop[2], filterTop[4]),Scalar(filterTop[1], filterTop[3], filterTop[5]),threshold0);
			inRange(cpu_HSV0,Scalar(filterBot[0], filterBot[2], filterBot[4]),Scalar(filterBot[1], filterBot[3], filterBot[5]),threshold1);
	
			// Search for paddle locations
			locatePaddle(topRegion,threshold0,frame0_warped);
			locatePaddle(botRegion,threshold1,frame0_warped);

		    toggle = 0;
		}

//-----------------------------------------------------------------------------------------------------------------
// Motion Detection
//-----------------------------------------------------------------------------------------------------------------
	
	// Compute the absolte value of the difference
	cv::cuda::absdiff(gpu_grayImage0, gpu_grayImage1, gpu_differenceImage);

	// Threshold the difference image
	cv::cuda::threshold(gpu_differenceImage, gpu_thresholdImage, 50, 255, cv::THRESH_BINARY);
	gpu_thresholdImage.download(result);

	// Find the location of any moving object and show the final frame
	if (toggle == 0) 
	{
        searchForMovement(result,frame0_warped);

		// Crop the image to the "displayRegion" set by X1_WINDOW, Y1_WINDOW...
		crop = frame0_warped(displayRegion);
	    imshow("Frame", crop);		
		crop = cpu_HSV0(displayRegion);
	    imshow("HSV", crop);
		
	}
	
	else 
	{
        searchForMovement(result,frame1_warped);
		
		// Crop the image to the "displayRegion" set by X1_WINDOW, Y1_WINDOW...
	    crop = frame1_warped(displayRegion);
	    imshow("Frame", crop);
		crop = cpu_HSV1(displayRegion);
	    imshow("HSV", crop);
	}


	// Move the windows so they are adjacent
	if (frame_count == 1) {
		int width = X2_WINDOW-X1_WINDOW+5;
		moveWindow("Frame", 10, 10);
		moveWindow("HSV"  , 10+width, 10);
	}

		// Show Frames
		//crop = cpu_HSV0(displayRegion);
		//imshow("HSV", crop);
		crop = frame0(displayRegion);
		imshow("Normal", crop);
		crop = threshold0(displayRegion);
		imshow("TOP", crop);
		crop = threshold1(displayRegion);
		imshow("BOTTOM", crop);

		frame_count++;
		waitKey(30);
	}
return 0;
}




