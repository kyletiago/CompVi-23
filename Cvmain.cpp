//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>

// Week 1 Exercise 1: Open an image
//using namespace cv;
//using namespace std;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/test.jpg");  //read the image "ar1.jpg" and store it in 'img', please the change the first argument to the absolute path of your image.
//	if (img.empty()) //check whether the image is loaded or not
//	{
//		cout << "Error : Image cannot be loaded..!!" << endl;
//		return -1;
//	}
//	namedWindow("image"); //create a window with the name "image"
//	imshow("image", img); //display the image which is stored in the 'img' in the "image" window
//	waitKey(0); //wait infinite time for a keypress
//	return 0;
//}

// Week 1 Exercise 2: Video File
//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;
//int main()
//{
//	VideoCapture capture("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/video.mp4"); // open the video file for reading, change number to zero for webcam
//	if (!capture.isOpened())  // if not success, exit program
//	{
//		cout << "Cannot open the video file" << endl;
//		return -1;
//	}
//	double fps = capture.get(cv::CAP_PROP_POS_FRAMES); //get the frames rate of the video
//	int delay = (int)(1000 / fps); //delay between two frames
//	Mat frame;
//	namedWindow("video");
//	int key = 0;
//	while (key != 27) // press "Esc" to stop
//	{
//		if (!capture.read(frame))
//		{
//			break;
//		}
//		imshow("video", frame);
//		key = waitKey(15);
//	}
//	return 0;
//}


// Week 2 Exercise 1: Return matrices
//using namespace cv;
//using namespace std;
//int main()
//{
//	Mat A(2, 4, CV_32FC1, 2);
//	cout << "A = " << endl << " " << A << endl << endl;
//	Mat B = Mat::eye(4, 4, CV_32FC1);
//	cout << "B = " << endl << " " << B << endl << endl;
//	Mat F = A.clone();
//	Mat G;
//	B.copyTo(G);
//	G.at<float>(2, 3) = 5;
//	cout << "F = " << endl << " " << F << endl << endl << "G = " << endl << " " << G << endl;
//	Mat M = F * G; //Matrix Multiplication
//	cout << "M = " << endl << " " << M << endl << endl;
//	Mat F_row = F.row(1);
//	G.row(2).copyTo(F_row); //replace the 2nd row of F by the 3rd row of G
//	cout << "F = " << endl << " " << F << endl << endl;
//	Mat N;
//	M.reshape(1, 1).copyTo(N);
//	cout << "N = " << endl << " " << N << endl << endl;
//	return 0;
//}

// Week 2 Exercise 2, mouse coordinate clicks
//#include<opencv2/imgproc/imgproc.hpp>
//using namespace std;
//using namespace cv;
//void locator(int event, int x, int y, int flags, void* userdata) { //function to track mouse movement and click//
//    if (event == EVENT_LBUTTONDOWN) { //when left button clicked//
//        cout << "Left click has been made, Position:(" << x << "," << y << ")" << endl;
//    }
//
//}
//int main() {
//    Mat image = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/test.jpg");//loading image in the matrix//
//    namedWindow("Track");//declaring window to show image//
//    setMouseCallback("Track", locator, NULL);//Mouse callback function on define window//
//    imshow("Track", image);//showing image on the window//
//    waitKey(0);//wait for keystroke//
//    return 0;
//}

// Week 2 Exercise 3: Write something press enter
//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>
//#include <vector>
//#include <opencv2/imgproc/imgproc.hpp>
//
//using namespace cv;
//using namespace std;
//vector<Point> capturePoint;
//void onMouse(int event, int x, int y, int flags, void* param)
//{
//	int n = 0;
//	switch (event)
//	{
//	case EVENT_LBUTTONDOWN: //click left button of mouse
//		cout << "Coordinate: (" << x << ',' << y << ')' << endl;
//		n++;
//		if (n < 2)
//			capturePoint.push_back(Point(x, y));
//		break;
//	}
//}
//int main()
//{
//	string text;
//	cin >> text; //input any characters and finish by press "Enter"
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/Leonardo.png");
//	namedWindow("image", WINDOW_AUTOSIZE);
//	setMouseCallback("image", onMouse, NULL);
//	imshow("image", img);
//	waitKey(0);
//	rectangle(img, capturePoint[0], capturePoint[1], Scalar(0, 0, 255)); //draw rectangle
//	putText(img, text, Point(capturePoint[0].x + 2, capturePoint[0].y - 10), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0)); // overlay text
//	imshow("image", img);
//	waitKey(0);
//	return 0;
//}

// Week 3 Exercise 1: Gaussian Filter with different kernel sizes
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//using namespace cv;
//int main()
//{
//	namedWindow("Original Image", WINDOW_AUTOSIZE);
//	namedWindow("Smoothed Image", WINDOW_AUTOSIZE);
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg");
//	imshow("Original Image", img);
//	Mat smooth_img;
//	char text[35];
//	for (int i = 5; i <= 21; i = i + 4)
//	{
//		_snprintf_s(text, 35, "Kernel Size : %d x %d", i, i);
//		GaussianBlur(img, smooth_img, Size(i, i), 0, 0);
//		putText(smooth_img, text, Point(img.cols / 4, img.rows / 8), cv::FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 0), 2);
//		imshow("Smoothed Image", smooth_img);
//		waitKey(0);
//	}
//	return 0;
//}

// Week 3 Exercise 2: Image to Grayscale
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//using namespace cv;
//using namespace std;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg");
//	namedWindow("Original Image", WINDOW_AUTOSIZE);
//	namedWindow("Sobel_X", WINDOW_AUTOSIZE);
//	namedWindow("Sobel_Y", WINDOW_AUTOSIZE);
//	namedWindow("Sobel", WINDOW_AUTOSIZE);
//	namedWindow("Laplacian", WINDOW_AUTOSIZE);
//	imshow("Original Image", img);
//	Mat smooth_img, gray_img;
//	GaussianBlur(img, smooth_img, Size(3, 3), 0, 0); //Gaussian smooth
//	cvtColor(smooth_img, gray_img, cv::COLOR_BGR2GRAY); //convert to gray-level image
//	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, SobelGrad;
//	Sobel(gray_img, grad_x, CV_32FC1, 1, 0);
//	convertScaleAbs(grad_x, abs_grad_x); //gradient X
//	imshow("Sobel_X", abs_grad_x);
//	Sobel(gray_img, grad_y, CV_32FC1, 0, 1);
//	convertScaleAbs(grad_y, abs_grad_y); //gradient Y
//	imshow("Sobel_Y", abs_grad_y);
//	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, SobelGrad); //total Sobel gradient
//	imshow("Sobel", SobelGrad);
//	Mat Lap, abs_Lap;
//	Laplacian(gray_img, Lap, CV_32FC1, 3);
//	convertScaleAbs(Lap, abs_Lap); //Laplacian operator
//	imshow("Laplacian", abs_Lap);
//	waitKey(0);
//	return 0;
//}

// Week 3 Exercise 3: Transform Image
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//using namespace cv;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg");
//	namedWindow("Original Image", WINDOW_AUTOSIZE);
//	imshow("Original Image", img);
//	namedWindow("Warped Image", WINDOW_AUTOSIZE);
//	namedWindow("Affine", WINDOW_AUTOSIZE);
//	int iAngle = 180;
//	createTrackbar("Angle", "Affine", &iAngle, 360);
//	int Percentage = 100;
//	double scale;
//	createTrackbar("Scale", "Affine", &Percentage, 200);
//	int iImageHieght = img.rows / 2;
//	int iImageWidth = img.cols / 2;
//	createTrackbar("XTranslation", "Affine", &iImageWidth, img.cols);
//	createTrackbar("YTranslation", "Affine", &iImageHieght, img.rows);
//	int key;
//	Mat imgwarped, matRotate, matTranslate, warpmat;
//	Mat increMat = (Mat_<double>(1, 3) << 0, 0, 1);
//	while (true)
//	{
//		scale = (double)Percentage / 100; //rescale
//		matRotate = getRotationMatrix2D(Point(iImageWidth, iImageHieght), (iAngle - 180), scale); //Rotate
//		matTranslate = (Mat_<double>(3, 1) << (double)(iImageWidth - img.cols / 2), (double)(iImageHieght - img.rows / 2), 1);//translate
//		warpmat = matRotate * matTranslate;
//		Mat R_col = matRotate.col(2);
//		warpmat.copyTo(R_col);
//		warpAffine(img, imgwarped, matRotate, img.size()); //warp image
//		imshow("Warped Image", imgwarped);
//		key = waitKey(30);
//		if (key == 27)
//		{
//			break;
//		}
//	}
//	return 0;
//}

//Week 4 Exercise 1: Change brightness and contrast of an image
//#include "opencv2/highgui/highgui.hpp"
//using namespace cv;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg");
//	Mat imgHB = img + Scalar(75, 75, 75); //increase the brightness by 75 units
//	 //img.convertTo(imgH, -1, 1, 75);
//	Mat imgLB = img + Scalar(-75, -75, -75); //decrease the brightness by 75 units
//	 //img.convertTo(imgL, -1, 1, -75);
//	Mat imgHC, imgLC;
//	img.convertTo(imgHC, -1, 2, 0); //increase the contrast (double)
//	img.convertTo(imgLC, -1, 0.5, 0); //decrease the contrast (halve)
//	namedWindow("Original Image", WINDOW_AUTOSIZE);
//	namedWindow("High Brightness", WINDOW_AUTOSIZE);
//	namedWindow("Low Brightness", WINDOW_AUTOSIZE);
//	namedWindow("High Contrast", WINDOW_AUTOSIZE);
//	namedWindow("Low Contrast", WINDOW_AUTOSIZE);
//	imshow("Original Image", img);
//	imshow("High Brightness", imgHB);
//	imshow("Low Brightness", imgLB);
//	imshow("High Contrast", imgHC);
//	imshow("Low Contrast", imgLC);
//	waitKey(0);
//	return 0;
//}

// Week 4 Exercise 2: Change brightness and constrast of a video
//#include "opencv2/highgui/highgui.hpp"
//using namespace cv;
//int main()
//{
//	VideoCapture capture("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/video.mp4");
//	namedWindow("Original Video", WINDOW_AUTOSIZE);
//	namedWindow("Brightness Increased", WINDOW_AUTOSIZE);
//	namedWindow("Contrast Increased", WINDOW_AUTOSIZE);
//	Mat frame, imgHB, imgHC;
//	int key = 0;
//	while (key != 27) // press "Esc" to stop
//	{
//		if (!capture.read(frame))
//		{
//			break;
//		}
//		imgHB = frame + Scalar(75, 75, 75); //increase the brightness by 75 units
//		frame.convertTo(imgHC, -1, 2, 0); //increase the contrast (double)
//		imshow("Original Video", frame);
//		imshow("Brightness Increased", imgHB);
//		imshow("Contrast Increased", imgHC);
//		key = waitKey(30);
//	}
//	return 0;
//}

// Week 4 Exercise 3: Use sliders to change brightness and contrast
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//using namespace cv;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg");
//	namedWindow("Original Image", WINDOW_AUTOSIZE);
//	imshow("Original Image", img);
//	namedWindow("Brightness", WINDOW_AUTOSIZE);
//	namedWindow("Contrast", WINDOW_AUTOSIZE);
//	int iBright = 255;
//	createTrackbar("Bright", "Brightness", &iBright, 510);
//	int iPercentage = 100;
//	createTrackbar("Percentage", "Contrast", &iPercentage, 200);
//	int key, Brightness;
//	float Contrast;
//	Mat imgB, imgC;
//	while (true)
//	{
//		Brightness = iBright - 255;
//		img.convertTo(imgB, -1, 1, Brightness);
//		Contrast = (float)iPercentage / 100;
//		img.convertTo(imgC, -1, Contrast, 0);
//		imshow("Brightness", imgB);
//		imshow("Contrast", imgC);
//		key = waitKey(30);
//		if (key == 27)
//		{
//			break;
//		}
//	}
//	return 0;
//}

// Week 5 Exercise 1: Gray-scale histogram calculation
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg", 0);
//	Mat grayHist;
//	int histSize = 256;
//	float range[] = { 0, 256 };
//	const float* histRange = { range };
//	calcHist(&img, 1, 0, Mat(), grayHist, 1, &histSize, &histRange); //calculate gray-scale histogram
//	cout << grayHist << endl; //display histogram 
//	return 0;
//}

// Week 5 Exercise 2: HoG extraction
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include <iostream>
//using namespace cv;
//using namespace std;
//int main()
//{
//	Mat img = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/building.jpg", 0);
//	Mat resized_img;
//	resize(img, resized_img, Size(64, 128)); //resize image
//	HOGDescriptor hog;
//	vector<float>descriptors;
//	hog.compute(resized_img, descriptors, Size(0, 0), Size(0, 0)); //compute HoG histogram
//	cout << descriptors.size() << endl;
//	return 0;
//}

// Week 5 Exercise 3: SIFT Feature Extraction
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/legacy/legacy.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;
int main()
{
	namedWindow("SIFT", WINDOW_AUTOSIZE);
	namedWindow("SIFTmatching", WINDOW_AUTOSIZE);
	Mat img1 = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/notre1.jpg", 0);
	Mat img2 = imread("C:/Users/Kyle/Documents/University/Masters/Computer Vision/database/notre2.jpg", 0);
	cv::SIFT sift1, sift2;
	int minHessian = 500;
	Ptr<SIFT> detector = SIFT::create(minHessian);
	vector<KeyPoint> key_points1, key_points2;
	Mat descriptors1, descriptors2, maskmat;
	detector->detectAndCompute(img1, maskmat, key_points1, descriptors1);
	detector->detectAndCompute(img2, maskmat, key_points2, descriptors2);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches); //keypoints matching
	std::nth_element(matches.begin(), matches.begin() + 99, matches.end()); //extract 100 best matches
	matches.erase(matches.begin() + 100, matches.end());  // delete the rest matches
	Mat keypoint_img, matching_img;
	drawKeypoints(img1, key_points1, keypoint_img, Scalar(255, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT", keypoint_img);
	drawMatches(img1, key_points1, img2, key_points2, matches, matching_img, Scalar(255, 255, 255), Scalar::all(-1), maskmat, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFTmatching", matching_img);
	waitKey(0);
	return 0;
}
