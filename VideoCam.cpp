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
	// VideoCapture cap("testfootage.mp4");
	VideoCapture cap(0);
	if(!cap.isOpened()) {
		cout<<"Unable to open camera";
		exit(-1);
	}

	Mat image;
	int FPS=24, iRows, iCols, ratio=2;
	cap>>image;
	// image = imread("im1.jpg", 1);
	cout<<"rows: "<<image.rows<<" cols: "<<image.cols<<" ch: "<<image.channels()<<" "<<image.isContinuous()<<endl;
	Mat smallImage(image.rows/ratio, image.cols/ratio, CV_8UC3, Scalar(43, 87, 12));
	Mat oldSmallImage(image.rows/ratio, image.cols/ratio, CV_8UC3, Scalar(43, 87, 12));
	Mat diff(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	Mat dx(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	Mat dy(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	// Mat outpImage(image.rows/ratio, image.cols/ratio, CV_64FC3, Scalar(43, 87, 12));
	iRows = image.rows;
	iCols = image.cols;

	while(true) {
		cap>>image;
		imshow("reading", image);
		if(image.empty()) {
			cout<<"Cant read from device";
		}
		if(cv::waitKey(1000.0/FPS) == 27) break;
	}

	int tp=2;
	while(true) {
		cap>>image;
		// if(tp == 1) {
		// 	image = imread("im1.jpg", 1);
		// }
		// else {
		// 	image = imread("im2.jpg", 1);
		// }
		imshow("reading", image);
		if(image.empty()) {
			cout<<"Cant read from device";
		}

		MatIterator_<Vec3b> og_it, og_end, sm_it, sm_end;
		int tp = 0;

		for( og_it = image.begin<Vec3b>(), og_end = image.end<Vec3b>(), sm_it = smallImage.begin<Vec3b>(), 
					sm_end = smallImage.end<Vec3b>(); og_it != og_end && sm_it != sm_end; og_it+=ratio, ++sm_it) {
			if(tp%(iCols/ratio) == 0 && tp!=0) {
				og_it+=iCols*(ratio-1);
			}
            (*sm_it)[0] = (*og_it)[0];
            (*sm_it)[1] = (*og_it)[1];
            (*sm_it)[2] = (*og_it)[2];
            tp++;
        }
        // fastNlMeansDenoisingColored(smallImage,smallImage,15,10,7,21);
        blur(smallImage, smallImage, Size(3, 3));

        Mat smallImage_64F, oldSmallImage_64F;
        smallImage.convertTo(smallImage_64F, CV_64FC3);
        oldSmallImage.convertTo(oldSmallImage_64F, CV_64FC3);

        diff = smallImage_64F - oldSmallImage_64F;

		Mat kern = (Mat_<char>(3,3) <<  1, 0,  -1,
		                                2, 0, -2,
		                                1, 0,  -1);
		filter2D(smallImage_64F, dx, smallImage_64F.depth(), kern);

		kern = (Mat_<char>(3,3) <<  -1, -2, -1,
	                                0, 0, 0,
	                                1, 2, 1);
		filter2D(smallImage_64F, dy, smallImage_64F.depth(), kern);

		int sqSize = 21;
		Mat tp1(sqSize*sqSize, 2, CV_64FC1, 1);
		Mat tp2(sqSize*sqSize, 1, CV_64FC1, 1);
		Mat motionXMat(diff.rows, diff.cols, CV_64FC1, 1);
		Mat motionYMat(diff.rows, diff.cols, CV_64FC1, 1);
		Mat motionMat = Mat::zeros(diff.rows, diff.cols, CV_64FC3);
		smallImage.copyTo(motionMat);

		smallImage.copyTo(oldSmallImage);

		Mat curGImage;
		cvtColor( smallImage, curGImage, CV_BGR2GRAY );
		Mat corners, showImage;
        goodFeaturesToTrack(curGImage, corners, 0, 0.05, 0.2, noArray(), 3, false, 0.04);
        for(int l=0;l<corners.rows;l++) {
        	float* curPoint = corners.ptr<float>(l);
        	int i = curPoint[1], j = curPoint[0]*3;

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
					// cout<<"he: "<<(int)curd[l]<<"  "<<(int)curtp2[0]<<endl;
					tp++;
				}
			}

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
			double len = sqrt(sec.ptr<double>(0)[0]*sec.ptr<double>(0)[0] + sec.ptr<double>(1)[0]*sec.ptr<double>(1)[0]);
			// if(len > 1 && len < 4) {
				double m = 0 + sec.ptr<double>(1)[0] / sec.ptr<double>(0)[0];
				// MyLine(motionMat, Point(j/3, i), Point(j/3 + 6*sec.ptr<double>(0)[0], (i - 6*sec.ptr<double>(1)[0])));
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
			// }
        }



		// tp = 0;
		// for(int i=5;i<diff.rows-5;i+=20) {
		// 	double* curMotX = motionXMat.ptr<double>(i);
		// 	double* curMotY = motionYMat.ptr<double>(i);
		// 	for(int j=9;j<3*diff.cols-10;j=j+60) {
		// 		tp=0;
		// 		for(int k=i-sqSize/2;k<i+1+sqSize/2;k++) {
		// 			double* curx = dx.ptr<double>(k);
		// 			double* cury = dy.ptr<double>(k);
		// 			double* curd = diff.ptr<double>(k);
		// 			for(int l=j-6;l<j+7;l=l+3) {

		// 				double* curtp1 = tp1.ptr<double>(tp);
		// 				double* curtp2 = tp2.ptr<double>(tp);
		// 				curtp1[0] = curx[l];
		// 				curtp1[1] = cury[l];
		// 				curtp2[0] = curd[l];
		// 				// cout<<"he: "<<(int)curd[l]<<"  "<<(int)curtp2[0]<<endl;
		// 				tp++;
		// 			}
		// 		}

		// 		Mat fir = (tp1.t() * tp1);
		// 		if(determinant(fir) <= 0.5) {
		// 			continue;
		// 		}
		// 		fir = (fir).inv(DECOMP_LU);
		// 		Mat sec = fir * tp1.t() * tp2;
		// 		curMotX[i] = sec.ptr<double>(0)[0];
		// 		curMotY[j/3] = sec.ptr<double>(1)[0];
		// 		// cout<<sec.ptr<double>(0)[0]<<endl;//<<"  "<<curMotY[j/3]<<endl;
		// 		// MyLine(motionMat, Point(i, j/3), Point(i + 3*sec.ptr<double>(0)[0], (j/3 + 3*sec.ptr<double>(1)[0])));
		// 		double len = sqrt(sec.ptr<double>(0)[0]*sec.ptr<double>(0)[0] + sec.ptr<double>(1)[0]*sec.ptr<double>(1)[0]);
		// 		if(len > 1 && len < 10) {
		// 			double m = 0 - sec.ptr<double>(1)[0] / sec.ptr<double>(0)[0];
		// 			// cout<<m<<endl;
		// 			// cout<<sec.ptr<double>(1)[0]<<endl<<endl<<endl;
		// 			// MyLine(motionMat, Point(j/3, i), Point(j/3 + 3*sec.ptr<double>(0)[0], (i - 3*sec.ptr<double>(1)[0])));
		// 			double p = -1/m;
		// 			MyLine(motionMat, Point(j/3 + 3/sqrt(1+p*p), i - 3*p/(sqrt(1+p*p))), Point(j/3 + 15*sec.ptr<double>(0)[0]/sqrt(1+m*m), i - 15*m*sec.ptr<double>(1)[0]/(sqrt(1+m*m))));
		// 			MyLine(motionMat, Point(j/3 - 3/sqrt(1+p*p), i + 3*p/(sqrt(1+p*p))), Point(j/3 + 15*sec.ptr<double>(0)[0]/sqrt(1+m*m), i - 15*m*sec.ptr<double>(1)[0]/(sqrt(1+m*m))));
		// 		}
		// 	}
		// }

		// cout<<"x "<<motionXMat<<endl;//+
		// cout<<"Y "<<motionYMat<<endl<<endl<<endl<<endl;//-

		// imshow("Camera feed diff", diff);
		// imshow("Camera feed dx", dx);
		// imshow("Camera feed dy", dy);
		imshow("Camera feed blur", smallImage);
		imshow("Camera feed motion", motionMat);
		// cout<<sec.t();

		smallImage.copyTo(oldSmallImage);
		if(cv::waitKey(1000.0/FPS) == 27) break;
	}
	// waitKey(0);
}