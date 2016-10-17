#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void MyLine( Mat img, Point start, Point end )
{
  int thickness = 2;
  int lineType = 8;
  line( img,
        start,
        end,
        Scalar( 20, 210, 25 ),
        1,
        8 );
}

int main() {
	//Use following line to read frames from video
	VideoCapture cap("test2.mp4");
	//Use following line to read frames from camera
	// VideoCapture cap(0);

	//Check for error in input method
	if(!cap.isOpened()) {
		cout<<"Unable to open camera";
		exit(-1);
	}

	Mat image;
	int FPS=24, iRows, iCols;

	//Downsize the input frames by this ratio.
	//Can be set to 1, 2, 4...
	int ratio=1;

	cap>>image;				//Read the first frame from input

	cout<<"rows: "<<image.rows<<" cols: "<<image.cols;
	cout<<" ch: "<<image.channels()<<" "<<image.isContinuous()<<endl;

	//Mat for storing the downsized images.
	Mat smallImage(image.rows/ratio, image.cols/ratio, CV_8UC3, Scalar(43, 87, 12));
	Mat oldSmallImage(image.rows/ratio, image.cols/ratio, CV_8UC3, Scalar(43, 87, 12));

	//Matrics for storing the values got after doing
	//the optic flow operations.
	Mat diff(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	Mat dx(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	Mat dy(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));

	iRows = image.rows;
	iCols = image.cols;

	//Do not start estimation of optic flow unless the
	//escape key is first pressed.
	while(true) {
		cap>>image;									//Read frames
		imshow("reading", image);					//Show the read frame
		if(image.empty()) {
			cout<<"Cant read from device";
		}
		if(cv::waitKey(1000.0/FPS) == 27) break;	//Check if escape pressed
	}

	// int tp=2;
	while(true) {
		cap>>image;									//Read frames

		imshow("reading", image);					//Show the read frame
		if(image.empty()) {
			cout<<"Cant read from device";
		}

		//Declare iterators for iterating over the original
		//and downsized mat of frames.
		MatIterator_<Vec3b> og_it, og_end, sm_it, sm_end;

		int tp = 0;

		//This is a custom implementation of the downsizing
		//of the original image. We just subsample the original image
		//and store it in the new Mat.
		for( og_it = image.begin<Vec3b>(), og_end = image.end<Vec3b>(), 
					sm_it = smallImage.begin<Vec3b>(), sm_end = smallImage.end<Vec3b>(); 
					og_it != og_end && sm_it != sm_end; og_it+=ratio, ++sm_it) {
			if(tp%(iCols/ratio) == 0 && tp!=0) {
				og_it+=iCols*(ratio-1);
			}
            (*sm_it)[0] = (*og_it)[0];
            (*sm_it)[1] = (*og_it)[1];
            (*sm_it)[2] = (*og_it)[2];
            tp++;
        }

        //Blur the downsized image inplace to reduce noise
        //due to subsampling and otherwise.
        blur(smallImage, smallImage, Size(3, 3));

        //Convert the Mat to use the float data type as
        //our calculations will involve decimal 
        //point numbers.
        Mat smallImage_64F, oldSmallImage_64F;
        smallImage.convertTo(smallImage_64F, CV_64FC3);
        oldSmallImage.convertTo(oldSmallImage_64F, CV_64FC3);

        //This is the first operation - taking the diff 
        //of current and previous frame.
        diff = smallImage_64F - oldSmallImage_64F;

        //Now we take the dx at every point in image. The 
        //following kernel is used for this purpose...
		Mat kern = (Mat_<char>(3,3) <<  1, 0,  -1,
		                                2, 0, -2,
		                                1, 0,  -1);
		filter2D(smallImage_64F, dx, smallImage_64F.depth(), kern);

		//Now we take the dy at every point in image. The 
        //following kernel is used for this purpose...
		kern = (Mat_<char>(3,3) <<  -1, -2, -1,
	                                0, 0, 0,
	                                1, 2, 1);
		filter2D(smallImage_64F, dy, smallImage_64F.depth(), kern);

		//We estimate optic flow in every image window
		//of size 21x21. The followin datastructures
		//are deckared for these operations.
		int sqSize = 21;
		Mat tp1(sqSize*sqSize, 2, CV_64FC1, 1);
		Mat tp2(sqSize*sqSize, 1, CV_64FC1, 1);
		Mat motionXMat(diff.rows, diff.cols, CV_64FC1, 1);
		Mat motionYMat(diff.rows, diff.cols, CV_64FC1, 1);
		Mat motionMat = Mat::zeros(diff.rows, diff.cols, CV_64FC3);

		smallImage.copyTo(motionMat);
		smallImage.copyTo(oldSmallImage);

		//Convert the image to grayscale to be able to use
		//it with imbuilt openCV functions.
		Mat curGImage;
		cvtColor( smallImage, curGImage, CV_BGR2GRAY );

		//Get corners in the current frame using the inbuilt
		//goodFeaturesToTrack function. We find optical flow 
		//only on these points. More details about this are 
		//available on my blog post on 
		// http://mayankrajoria.com/blog
		Mat corners, showImage;
        goodFeaturesToTrack(curGImage, corners, 0, 0.05, 0.02, noArray(), 3, false, 0.04);

        for(int l=0;l<corners.rows;l++) {
        	float* curPoint = corners.ptr<float>(l);
        	int i = curPoint[1], j = curPoint[0]*3;
        	if(i<15) continue;
        	tp=0;
        	for(int k=i-sqSize/2;k<i+1+sqSize/2;k++) {
				double* curx = dx.ptr<double>(k);
				double* cury = dy.ptr<double>(k);
				double* curd = diff.ptr<double>(k);
				for(int l=j-30;l<j+31;l=l+3) {

					double* curtp1 = tp1.ptr<double>(tp);
					double* curtp2 = tp2.ptr<double>(tp);
					curtp1[0] = curx[l];
					curtp1[1] = cury[l];
					curtp2[0] = curd[l];
					tp++;
				}
			}

			//Do th eoptic flow calculation for
			//the current window and also apply
			//non linear repeated estimation.
			Mat fir = (tp1.t() * tp1);
			if(determinant(fir) <= 0.5) {
				continue;
			}
			fir = (fir).inv(DECOMP_LU);
			Mat sec = fir * tp1.t();
			Mat pre;
			sec.copyTo(pre);
			sec = sec * tp2;
			int it = 3;
			while(it--) {
				Mat rb = tp1 * sec - tp2;
				sec = sec - pre * rb;
			}

			//Magnitude of the optic flow claculated 
			//in current window.
			double len = sqrt(sec.ptr<double>(0)[0]*sec.ptr<double>(0)[0] + sec.ptr<double>(1)[0]*sec.ptr<double>(1)[0]);

			//If magnitude is more than threshold, plot an
			//arrow for this flow in the image.
			if(len > 0.05) {
				double m = 0 + sec.ptr<double>(1)[0] / sec.ptr<double>(0)[0];
				double p = -1/m;
				MyLine(motionMat, Point(j/3, i), Point(j/3, i));
				if(sec.ptr<double>(0)[0] > 0) {
					MyLine(motionMat, Point(j/3 + 3/sqrt(1+p*p), i - 3*p/(sqrt(1+p*p))), Point(j/3 + 25*len/sqrt(1+m*m), i - 25*m*len/(sqrt(1+m*m))));
					MyLine(motionMat, Point(j/3 - 3/sqrt(1+p*p), i + 3*p/(sqrt(1+p*p))), Point(j/3 + 25*len/sqrt(1+m*m), i - 25*m*len/(sqrt(1+m*m))));
				}
				else {
					MyLine(motionMat, Point(j/3 + 3/sqrt(1+p*p), i - 3*p/(sqrt(1+p*p))), Point(j/3 - 25*len/sqrt(1+m*m), i + 25*m*len/(sqrt(1+m*m))));
					MyLine(motionMat, Point(j/3 - 3/sqrt(1+p*p), i + 3*p/(sqrt(1+p*p))), Point(j/3 - 25*len/sqrt(1+m*m), i + 25*m*len/(sqrt(1+m*m))));
				}
			}
        }


        //Show some outputs of the blurred image
        //and the other with optic flow arrows overlayed
        //onto it.
		imshow("Camera feed blur", smallImage);
		imshow("Camera feed motion", motionMat);

		//Store the current frame as the old frame.
		smallImage.copyTo(oldSmallImage);

		//Listen to press of escape key and quit
		//if the key is pressed.
		if(cv::waitKey(1000.0/FPS) == 27) break;
	}
	// waitKey(0);
}