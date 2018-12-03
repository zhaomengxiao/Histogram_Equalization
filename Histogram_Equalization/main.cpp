#include <opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>


using namespace std;
using namespace cv;

void plotHistogram(String windowname,Mat src) {
	MatND dstHist;//�õ���ֱ��ͼ     
	int dims = 1;//�õ���ֱ��ͼ��ά�� �Ҷ�ͼ��ά��Ϊ1
	float hranges[2] = { 0, 255 };
	const float *ranges[1] = { hranges };   // ������ҪΪconst���ͣ���ά��������ָ��ÿ������ķ�Χ  
	int size = 256;//ֱ��ͼ������������� �������걻�ֳ�256��
	int channels = 0;//ͼ���ͨ�� �Ҷ�ͼ��ͨ����Ϊ0
	//����ͼ���ֱ��ͼ  
	calcHist(&src, 1, &channels, Mat(), dstHist, dims, &size, ranges);
	int scale = 1;
	Mat dstImage(size * scale, size, CV_8U, Scalar(0));
	//��ȡ���ֵ����Сֵ  
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0); //�ҵ�ֱ��ͼ�е����ֵ����Сֵ 
	//���Ƴ�ֱ��ͼ  
	int hpt = saturate_cast<int>(0.9 * size);//��ֹ���
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		line(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
	}
	imshow(windowname, dstImage);
}


int main(int argc, char* argv[]) 
{
	if (argc != 2)
	{
		cout << "����·��";
		return -1;
	}

	stringstream ss;

	string filename;
	ss.clear();
	ss << argv[1];
	
	if (!ss.good()){
		cout << "����·��" << endl;
		return -1;
	}
	ss >> filename;
	cout << filename << endl;
	
	Mat src = imread(filename, IMREAD_GRAYSCALE);
	if (src.empty()){
		cout << "cant load image" << endl;
		return -1;
	}
	imshow("src", src);
	



	//Histogram_Equalization
	//opencv
	Mat tmp_cv;
	src.copyTo(tmp_cv);
	equalizeHist(tmp_cv, tmp_cv);
	//myself
	Mat tmp;
	src.copyTo(tmp);
	int numofPixel = tmp.total();
	//ͳ�Ƹ���������
	vector<int>grayLevel(256, 0);	
	for (int i = 0; i < tmp.total();++i)
		grayLevel[tmp.data[i]]++;
	for (int i = 1; i < 256; ++i)
		grayLevel[i]+= grayLevel[i-1];
	//���½���ӳ���ϵ
	vector<int>grayLevel_mapped(256, 0);
	for (int i = 0; i < 256; ++i) {
		grayLevel_mapped[i] = double(grayLevel[i]) / numofPixel * 255;
	}
	for (int i = 0; i < tmp.total(); ++i)
		tmp.data[i] = grayLevel_mapped[tmp.data[i]];
	imshow("src", src);
	imshow("Histogram_Equalization", tmp);
	imshow("Histogram_Equalization_cv", tmp_cv);

	plotHistogram("src_h", src);
	plotHistogram("Eq_h", tmp);
	plotHistogram("cv_Eq_h", tmp_cv);

	

	waitKey();
	return 0;
}