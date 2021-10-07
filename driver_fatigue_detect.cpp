/*
* @File_name:  driver_fatigue_detect.cpp
* @Description: 利用dlib&opencv进行眼口闭开和低头检测，实现疲劳判断
* @Date:   2021-10-7 23:00:00
* @Author: @Tongji
*/

//#include <opencv2\opencv.hpp>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing.h>
#include <dlib\opencv\cv_image_abstract.h>
#include <dlib\pixel.h>
#include <dlib\opencv\cv_image.h>
#include <iostream>
/*姿态估计头文件*/
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>


using namespace std;
using namespace dlib;
using namespace cv;

int main()
{
	/**********变量定义和初始化**********/
	unsigned int blink_cnt = 0;			   //眨眼计数
	unsigned int open_mou_cnt = 0;         //张嘴计数
	unsigned int close_eye_cnt = 0;        //闭眼计数

	//眨眼过程：EAR>0.2-- - EAR<0.2-- - EAR>0.2
	float blink_EAR_before = 0.0;		//眨眼前
	float blink_EAR_now = 0.2;			//眨眼中
	float blink_EAR_after = 0.0;		//眨眼后

	//闭眼最大时长：EAR<0.2的持续时间
	unsigned int eye_close_duration = 1;  //闭眼时长
	unsigned int real_yawn = 1;  //张嘴时长 done!
	unsigned int detect_no_face_duration = 1; //未见人脸时长 done!

	//张嘴：MAR>0.5
	float MAR_THRESH = 0.5;

	//低头检测
	unsigned int nod_cnt = 0; //点头计数
	unsigned int nod_total = 0; //瞌睡点头
	int head_thresh = 8; //低头欧拉角（head）阈值
	
	//相机坐标系
	double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002,2.3950000000000000e+002, 0.0, 0.0, 1.0 };
	//图像中心坐标系
	double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

	//像素坐标系(xy)：填写凸轮的本征和畸变系数
	Mat cam_matrix = Mat(3, 3, CV_64FC1, K);
	Mat dist_coeffs = Mat(5, 1, CV_64FC1, D);

	//# 世界坐标系(UVW)：填写3D参考点--->14点

	std::vector<Point3d> object_pts;
	object_pts.push_back(Point3d(6.825897, 6.760612, 4.402142));     //#1 左眉左
	object_pts.push_back(Point3d(1.330353, 7.122144, 6.903745));     //#2 左眉右
	object_pts.push_back(Point3d(-1.330353, 7.122144, 6.903745));    //#3 右眉左
	object_pts.push_back(Point3d(-6.825897, 6.760612, 4.402142));    //#4 右眉右
	object_pts.push_back(Point3d(5.311432, 5.485328, 3.987654));     //#7 左眼左
	object_pts.push_back(Point3d(1.789930, 5.393625, 4.413414));     //#10 左眼右
	object_pts.push_back(Point3d(-1.789930, 5.393625, 4.413414));    //#13 右眼左
	object_pts.push_back(Point3d(-5.311432, 5.485328, 3.987654));    //#16 右眼右
	object_pts.push_back(Point3d(2.005628, 1.409845, 6.165652));     //#5 鼻左
	object_pts.push_back(Point3d(-2.005628, 1.409845, 6.165652));    //#6 鼻右
	object_pts.push_back(Point3d(2.774015, -2.080775, 5.048531));    //#19 嘴左
	object_pts.push_back(Point3d(-2.774015, -2.080775, 5.048531));   //#22 嘴右
	object_pts.push_back(Point3d(0.000000, -3.116408, 6.097667));    //#24 嘴下中间
	object_pts.push_back(Point3d(0.000000, -7.415691, 4.070434));    //#0 下巴

	//二维参考点（图像坐标），参考检测到的面部特征
	std::vector<Point2d> image_pts;

	//result
	Mat rotation_vec;                           //3 x 1
	Mat rotation_mat;                           //3 x 3 R
	Mat translation_vec;                        //3 x 1 T
	Mat pose_mat = Mat(3, 4, CV_64FC1);     //3 x 4 R | T
	Mat euler_angle = Mat(3, 1, CV_64FC1); //欧拉角矩阵

	//重新投影3D点的世界坐标轴以验证结果姿势
	std::vector<cv::Point3d> reprojectsrc;
	reprojectsrc.push_back(Point3d(10.0, 10.0, 10.0));
	reprojectsrc.push_back(Point3d(10.0, 10.0, -10.0));
	reprojectsrc.push_back(Point3d(10.0, -10.0, -10.0));
	reprojectsrc.push_back(Point3d(10.0, -10.0, 10.0));
	reprojectsrc.push_back(Point3d(-10.0, 10.0, 10.0));
	reprojectsrc.push_back(Point3d(-10.0, 10.0, -10.0));
	reprojectsrc.push_back(Point3d(-10.0, -10.0, -10.0));
	reprojectsrc.push_back(Point3d(-10.0, -10.0, 10.0));

	//重新投影的 2D 点
	std::vector<cv::Point2d> reprojectdst;
	reprojectdst.resize(8);

	//用于分解ProjectionMatrix()投影矩阵（）的临时缓冲区
	Mat out_intrinsics = Mat(3, 3, CV_64FC1);
	Mat out_rotation = Mat(3, 3, CV_64FC1);
	Mat out_translation = Mat(3, 1, CV_64FC1);


	/**********变量定义和初始化**********/


	/**********载入dlib face_landmark预测模型（26点）**********/
	frontal_face_detector face_detector = get_frontal_face_detector();
	shape_predictor pre_model_26;
	deserialize("E:\\Fatigue\\driver_fatigue_detect\\driver_fatigue_detect\\26_predictor.dat") >> pre_model_26;
	/**********载入dlib face_landmark预测模型（26点）**********/

	try 
	{
		VideoCapture cap(0);
		if (!cap.isOpened()) 
		{
			cout<<"未打开摄像头"<<endl;
			return 1;
		}

		double fps;
		double t = 0;

		while (waitKey(30) != 27)//30ms用户未按下ESC键,与帧率有关
		{
			Mat src;
			cap >> src;

			t = (double)getTickCount();

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
					shapes.emplace_back(pre_model_26(img, faces[i]));
				}
				
				if (!shapes.empty())
				{
					/**********26特征点**********/
					int faceNumber = shapes.size();
					for (int j = 0; j < faceNumber; j++)
					{
						for (int i = 0; i < 26; i++)
						{
							//cv::circle(src, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 0.5, cv::Scalar(255, 0, 0), -1);
							cv::putText(src, to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 255, 0));
						}
					}

					//填写二维参考点--->14点,有待优化
					image_pts.push_back(Point2d(shapes[0].part(1).x(), shapes[0].part(1).y())); //#1 左眉左
					image_pts.push_back(Point2d(shapes[0].part(2).x(), shapes[0].part(2).y())); //#2 左眉右
					image_pts.push_back(Point2d(shapes[0].part(3).x(), shapes[0].part(3).y())); //#3 右眉左
					image_pts.push_back(Point2d(shapes[0].part(4).x(), shapes[0].part(4).y())); //#4 右眉右

					image_pts.push_back(Point2d(shapes[0].part(7).x(), shapes[0].part(7).y())); //#7 左眼左
					image_pts.push_back(Point2d(shapes[0].part(10).x(), shapes[0].part(10).y())); //#10 左眼右
					image_pts.push_back(Point2d(shapes[0].part(13).x(), shapes[0].part(13).y())); //#13 右眼左
					image_pts.push_back(Point2d(shapes[0].part(16).x(), shapes[0].part(16).y())); //#16 右眼右

					image_pts.push_back(Point2d(shapes[0].part(5).x(), shapes[0].part(5).y())); //#5 鼻左
					image_pts.push_back(Point2d(shapes[0].part(6).x(), shapes[0].part(6).y())); //#6 鼻右

					image_pts.push_back(Point2d(shapes[0].part(19).x(), shapes[0].part(19).y())); //#19 嘴左
					image_pts.push_back(Point2d(shapes[0].part(22).x(), shapes[0].part(22).y())); //#22 嘴右
					image_pts.push_back(Point2d(shapes[0].part(24).x(), shapes[0].part(24).y())); //#24 嘴下中间
					image_pts.push_back(Point2d(shapes[0].part(0).x(), shapes[0].part(0).y()));   //#0 下巴

					/**********26特征点**********/

					/**********眼口坐标*********/
					//左眼坐标：6点
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

					unsigned int x_12 = shapes[0].part(12).x();
					unsigned int y_12 = shapes[0].part(12).y();

					//右眼坐标：6点
					unsigned int x_13 = shapes[0].part(13).x();
					unsigned int y_13 = shapes[0].part(13).y();

					unsigned int x_14 = shapes[0].part(14).x();
					unsigned int y_14 = shapes[0].part(14).y();

					unsigned int x_15 = shapes[0].part(15).x();
					unsigned int y_15 = shapes[0].part(15).y();

					unsigned int x_16 = shapes[0].part(16).x();
					unsigned int y_16 = shapes[0].part(16).y();

					unsigned int x_17 = shapes[0].part(17).x();
					unsigned int y_17 = shapes[0].part(17).y();

					unsigned int x_18 = shapes[0].part(18).x();
					unsigned int y_18 = shapes[0].part(18).y();

					//嘴巴坐标:6点
					unsigned int x_19 = shapes[0].part(19).x();
					unsigned int y_19 = shapes[0].part(19).y();

					unsigned int x_22 = shapes[0].part(22).x();
					unsigned int y_22 = shapes[0].part(22).y();

					unsigned int x_20 = shapes[0].part(20).x();
					unsigned int y_20 = shapes[0].part(20).y();

					unsigned int x_25 = shapes[0].part(25).x();
					unsigned int y_25 = shapes[0].part(25).y();

					unsigned int x_21 = shapes[0].part(21).x();
					unsigned int y_21 = shapes[0].part(21).y();

					unsigned int x_23 = shapes[0].part(23).x();
					unsigned int y_23 = shapes[0].part(23).y();
					/**********眼口坐标*********/

					/**********EAR*********/
					float height_left_eye = (y_11 - y_9 + y_12 - y_8) / 2;		//左眼纵高
					unsigned int length_left_eye = x_10 - x_7;   //左眼横宽


					if (height_left_eye == 0)  //当眼睛闭合的时候，纵高为0，此时重新赋值为1
						height_left_eye = 1;
					float EAR_left_eye;			//左眼宽高比
					EAR_left_eye = height_left_eye / length_left_eye;

					float height_right_eye = (y_17 - y_15 + y_18 - y_14) / 2;		//右眼纵高
					unsigned int length_right_eye = x_16 - x_13;     //右眼横宽
					if (height_right_eye == 0)  //当眼睛闭合的时候，纵高为0，此时重新赋值为1
						height_right_eye = 1;
					float EAR_right_eye;			//右眼宽高比
					EAR_right_eye = height_right_eye / length_right_eye;

					//EAR
					EAR_eyes = (EAR_left_eye + EAR_right_eye) / 2;
					/**********EAR*********/

					/**********MAR*********/
					unsigned int lenght_mouth = x_22 - x_19;			//嘴巴横宽
					float height_mouth = (y_25 - y_20 + y_23 - y_21) / 2;//嘴巴纵高
					
					MAR_mouth = height_mouth / lenght_mouth;//嘴巴的宽高比
					/**********MAR*********/

					//姿势计算：3D参考点+2D特征点
					solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

					//重新规划 
					projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

					//calc euler angle 角度计算
					Rodrigues(rotation_vec, rotation_mat);
					hconcat(rotation_mat, translation_vec, pose_mat);
					decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, noArray(), noArray(), noArray(), euler_angle);

					float HEAD_THRESH = 0.3;
					double head = euler_angle.at<double>(0);

					/**********低头检测(辅助)**********/
					if (head > head_thresh)
					{
						nod_cnt += 1;
					}
					else //如果连续3次都小于阈值，则表示瞌睡点头一次
					{
						if (nod_cnt >= 3)
						{
							nod_total += 1;
							nod_cnt = 0;
							if (nod_total > 10)
							{
								cout << "检测到多次低头行为，您已疲劳，请休息！" << endl;
								nod_total = 0;
							}
						}
					}

					char nod_total_text[30];
					_gcvt_s(nod_total_text, nod_total, 10);
					cout << "低头次数：" << nod_total_text << endl<<endl;

					//char head_text[30];
					//_gcvt_s(head_text, head, 10);
					//cout << "heading角:" << head_text << endl<<endl;

					image_pts.clear();

					/**********低头检测(辅助)**********/


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

					//cout << "EAR： " << EAR_eyes_text << endl << endl;
					//cout << "MAR： " << MAR_mouth_text << endl << endl;
					cout << "当前闭眼过程眨眼计数： " << blink_cnt_text << endl << endl;
					cout << "当前哈欠过程张嘴计数： " << open_mou_cnt_text << endl << endl;

					//cv::putText(src, string("FPS: ") + to_string(1 / (flame_end - flame_start)), Point(10, 350), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 0), 1, LINE_AA);

				}

			}
			/**********检测到人脸**********/

			/*视频帧率显示*/
			t = ((double)getTickCount() - t) / getTickFrequency();
			fps = 1.0 / t;
			putText(src, "FPS: "+to_string(fps), cvPoint(5, 20), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 255, 0));
			/*视频帧率显示*/

			//视频流显示
			cv::imshow("驾驶疲劳检测中...", src);


			//检测时间窗口
			clock_t start = clock();
			clock_t finish = clock();
			double consumeTime = (double)(finish - start);


			///**********哈欠行为警告**********/
			if ((open_mou_cnt / consumeTime) > 60)//张嘴频率大于60次/秒视为一次哈欠，数字待考证
			{
				    
					if (MAR_mouth < MAR_THRESH)//闭嘴时刻
					{
						open_mou_cnt = 0;
						real_yawn++;
						if (real_yawn > 8)
						{
							cout << "检测到多次哈欠行为，您已疲劳，请休息！" << endl;
							//for (int i = 0; i < 3; i++) {
							//	cout << '\a' << endl;
							//	Sleep(500);
							//}
							real_yawn = 0;
						}
					}
					else
					{
						open_mou_cnt++;
						cout << "检测到哈欠行为，当前哈欠次数：" << real_yawn << endl;

					}
			}
			/**********哈欠行为警告**********/

			/**********闭眼行为警告**********/
			if ((blink_cnt/ consumeTime) >60)//眨眼频率大于次60/秒视为一次闭眼过程，数字待考证
			{
				if (EAR_eyes > 0.18)//睁眼时刻
				{
					blink_cnt = 0;
					eye_close_duration++;
					if (eye_close_duration > 8)
					{
						cout << "检测到多次闭眼行为，您已疲劳，请休息！" << endl;
						//for (int i = 0; i < 3; i++) {
						//	cout << '\a' << endl; 
						//	Sleep(500);
						//}
						eye_close_duration = 0;
					}
				}
				else
				{
					blink_cnt++;
					if (blink_cnt > 15)
					{
						cout << "检测到闭眼行为，当前闭眼次数：" << eye_close_duration << endl;
					}
				}
			}
			///**********闭眼行为警告**********/

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
