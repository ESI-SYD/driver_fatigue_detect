/*
* @File_name:  driver_fatigue_detect.cpp
* @Description: 利用dlib&opencv开源视觉库进行眼口闭开检测，实现疲劳判断
* @Date:   2021-9-11 14:57:01
* @Author: SYD@TongJI
*/

#include <dlib\opencv.h>
#include <opencv2\opencv.hpp>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>

using namespace std;
using namespace dlib;
using namespace cv;

int main()
{
	/**********变量定义和初始化**********/
	unsigned int blink_cnt = 0;			//眨眼计数
	unsigned int open_mou_cnt = 0;         //张嘴计数
	unsigned int close_eye_cnt = 0;        //闭眼计数

	//眨眼过程：EAR>0.2-- - EAR<0.2-- - EAR>0.2
	float blink_EAR_before = 0.0;		//眨眼前
	float blink_EAR_now = 0.2;			//眨眼中
	float blink_EAR_after = 0.0;		//眨眼后

	//闭眼最大时长：EAR<0.2的持续时间，待更新
	unsigned int eye_close_duration = 0;  //闭眼时长
	unsigned int real_yawn = 1;  //张嘴时长
	unsigned int detect_no_face_duration = 1; //未见人脸时长 done!

	//张嘴：MAR>0.5
	float MAR_THRESH = 0.5;
	/**********变量定义和初始化**********/


	/**********载入dlib face_landmark预测模型（18点）**********/
	frontal_face_detector face_detector = get_frontal_face_detector();
	shape_predictor pre_modle_18;
	deserialize("E:\\Fatigue\\driver_fatigue_detect\\driver_fatigue_detect\\18_predictor.dat") >> pre_modle_18;
	/**********载入dlib face_landmark预测模型（18点）**********/

	try 
	{
		VideoCapture cap(0);
		if (!cap.isOpened()) 
		{
			cout<<"未打开摄像头"<<endl;
			return 1;
		}

		while (waitKey(30) != 27)//30ms用户未按下ESC键
		{
			Mat src;
			cap >> src;

			clock_t flame_start = clock();

			//将src转化为BGR
			cv_image<bgr_pixel> img(src);

			std::vector<dlib::rectangle> faces = face_detector(img);
			std::vector<full_object_detection> shapes;

			unsigned int faceNumber = faces.size();   //容器容量即人脸个数

			float MAR_mouth;
			float EAR_eyes;

			/**********未检测到人脸**********/
			if (faceNumber == 0)
			{
				if (detect_no_face_duration != 100)
				{
					cout << "未检测到人脸!!\t" <<"time: "<< detect_no_face_duration++<<endl;
					
				}
				else
				{
					cout << "较长时间未检测到人脸，判定疲劳！" << endl;
					detect_no_face_duration = 1;
				}
			}
			/**********未检测到人脸**********/

			/**********检测到人脸**********/
			else
			{
				detect_no_face_duration = 0;
				for (unsigned int i = 0; i < faceNumber; ++i)
				{
					shapes.emplace_back(pre_modle_18(img, faces[i]));
				}
				
				if (!shapes.empty())
				{
					/**********18特征点**********/
					int faceNumber = shapes.size();
					for (int j = 0; j < faceNumber; j++)
					{
						for (int i = 0; i < 18; i++)
						{
							//画18特征点
							cv::circle(src, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 0.5, cv::Scalar(255, 255, 0), -1);
							cv::putText(src, to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
						}
					}

					clock_t flame_end = clock();
					/**********18特征点**********/

					/**********眼口坐标*********/
					//左眼坐标：6点
					unsigned int x_0 = shapes[0].part(0).x();
					unsigned int y_0 = shapes[0].part(0).y();

					unsigned int x_1 = shapes[0].part(1).x();
					unsigned int y_1 = shapes[0].part(1).y();

					unsigned int x_2 = shapes[0].part(2).x();
					unsigned int y_2 = shapes[0].part(2).y();

					unsigned int x_3 = shapes[0].part(3).x();
					unsigned int y_3 = shapes[0].part(3).y();

					unsigned int x_4 = shapes[0].part(4).x();
					unsigned int y_4 = shapes[0].part(4).y();

					unsigned int x_5 = shapes[0].part(5).x();
					unsigned int y_5 = shapes[0].part(5).y();

					//右眼坐标：6点
					unsigned int x_6 = shapes[0].part(6).x();
					unsigned int y_6 = shapes[0].part(6).y();

					unsigned int x_7 = shapes[0].part(7).x();
					unsigned int y_7 = shapes[0].part(7).y();

					unsigned int x_8 = shapes[0].part(8).x();
					unsigned int y_8 = shapes[0].part(8).y();

					unsigned int x_9 = shapes[0].part(9).x();
					unsigned int y_9 = shapes[0].part(9).y();

					unsigned int x_10 = shapes[0].part(10).x();
					unsigned int y_10 = shapes[0].part(10).y();

					unsigned int x_11 = shapes[0].part(11).x();
					unsigned int y_11 = shapes[0].part(11).y();

					//嘴巴坐标:6点
					unsigned int x_12 = shapes[0].part(12).x();
					unsigned int y_12 = shapes[0].part(12).y();

					unsigned int x_15 = shapes[0].part(15).x();
					unsigned int y_15 = shapes[0].part(15).y();

					unsigned int x_13 = shapes[0].part(13).x();
					unsigned int y_13 = shapes[0].part(13).y();

					unsigned int x_17 = shapes[0].part(17).x();
					unsigned int y_17 = shapes[0].part(17).y();

					unsigned int x_14 = shapes[0].part(14).x();
					unsigned int y_14 = shapes[0].part(14).y();

					unsigned int x_16 = shapes[0].part(16).x();
					unsigned int y_16 = shapes[0].part(16).y();
					/**********眼口坐标*********/

					/**********EAR*********/
					float height_left_eye = (y_4 - y_2 + y_5 - y_1) / 2;		//左眼纵高
					unsigned int length_left_eye = x_3 - x_0;   //左眼横宽


					if (height_left_eye == 0)  //当眼睛闭合的时候，纵高为0，此时重新赋值为1
						height_left_eye = 1;
					float EAR_left_eye;			//左眼宽高比
					EAR_left_eye = height_left_eye / length_left_eye;

					float height_right_eye = (y_10 - y_8 + y_11 - y_7) / 2;		//右眼纵高
					unsigned int length_right_eye = x_9 - x_6;     //右眼横宽
					if (height_right_eye == 0)  //当眼睛闭合的时候，纵高为0，此时重新赋值为1
						height_right_eye = 1;
					float EAR_right_eye;			//右眼宽高比
					EAR_right_eye = height_right_eye / length_right_eye;

					//EAR
					EAR_eyes = (EAR_left_eye + EAR_right_eye) / 2;
					/**********EAR*********/

					/**********MAR*********/
					unsigned int lenght_mouth = x_15 - x_12;			//嘴巴横宽
					float height_mouth = (y_17 - y_13 + y_16 - y_14) / 2;//嘴巴纵高
					
					MAR_mouth = height_mouth / lenght_mouth;//嘴巴的宽高比
					/**********MAR*********/



					//张嘴计数
					if (MAR_mouth > MAR_THRESH)//阈值0.5
					{
						open_mou_cnt++;
					}

					//闭眼计数
					if (EAR_eyes < 0.2)
					{
						close_eye_cnt++;
					}

					//眨眼计数
					if (blink_EAR_now > EAR_eyes)
					{
						blink_EAR_now = EAR_eyes;
					}

					if (blink_EAR_now <= 0.2)
					{
						blink_cnt++;
						blink_EAR_before = 0.0;
						blink_EAR_now = 0.2;
						blink_EAR_after = 0.0;
					}


					char blink_cnt_text[30];   //眨眼计数
					char open_mou_cnt_text[30];   //张嘴计数
					char EAR_eyes_text[30];   //EAR
					char MAR_mouth_text[30];  //MAR

					_gcvt_s(EAR_eyes_text, EAR_eyes, 10);
					_gcvt_s(blink_cnt_text, blink_cnt, 10);
					_gcvt_s(MAR_mouth_text, MAR_mouth, 10);
					_gcvt_s(open_mou_cnt_text, open_mou_cnt, 10);

					cout << "EAR： " << EAR_eyes_text << endl << endl;
					cout << "MAR： " << MAR_mouth_text << endl << endl;
					cout << "眨眼计数： " << blink_cnt_text << endl << endl;
					cout << "张嘴计数： " << open_mou_cnt_text << endl << endl;

					//cv::putText(src, string("FPS: ") + to_string(1 / (flame_end - flame_start)), Point(10, 350), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 0), 1, LINE_AA);

				}

			}
			/**********检测到人脸**********/

			//视频流显示
			cv::imshow("驾驶疲劳检测中...", src);


			//检测时间窗口
			clock_t start = clock();
			clock_t finish = clock();
			double consumeTime = (double)(finish - start);


			/**********哈欠行为警告**********/
			if ((open_mou_cnt / consumeTime) > 60)//张嘴频率大于60次/秒视为一次哈欠，数字待考证
			{
				    
					if (MAR_mouth < MAR_THRESH)//闭嘴时刻
					{
						open_mou_cnt = 0;
						real_yawn++;
					}
					else
					{
						open_mou_cnt++;
						cout << "检测到哈欠行为，当前哈欠次数：" << real_yawn << endl;
					}
			}
			/**********哈欠行为警告**********/
			
			//持续闭眼警告，有较大瑕疵，待修复...
			if ((close_eye_cnt / consumeTime) > 60000)//眨眼频率大于60次/秒视为一次闭眼，数字待考证
			{
				if (EAR_eyes > 0.25)//睁眼时刻
				{
					close_eye_cnt = 0;
					eye_close_duration++;
					EAR_eyes = 0;//归零防闭眼误判断
				}
				else
				{
					close_eye_cnt++;
					cout << "检测到闭眼行为，当前闭眼次数：" << eye_close_duration << endl;
				}
			}

		}

	}
	catch (serialization_error& e)
	{
		cout << "dlib 预测模型未成功加载..." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}

}
