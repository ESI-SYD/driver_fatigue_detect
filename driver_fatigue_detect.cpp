/*
* @File_name:  driver_fatigue_detect.cpp
* @Description: ����dlib&opencv��Դ�Ӿ�������ۿڱտ���⣬ʵ��ƣ���ж�
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
	/**********��������ͳ�ʼ��**********/
	unsigned int blink_cnt = 0;			//գ�ۼ���
	unsigned int open_mou_cnt = 0;         //�������
	unsigned int close_eye_cnt = 0;        //���ۼ���

	//գ�۹��̣�EAR>0.2-- - EAR<0.2-- - EAR>0.2
	float blink_EAR_before = 0.0;		//գ��ǰ
	float blink_EAR_now = 0.2;			//գ����
	float blink_EAR_after = 0.0;		//գ�ۺ�

	//�������ʱ����EAR<0.2�ĳ���ʱ�䣬������
	unsigned int eye_close_duration = 0;  //����ʱ��
	unsigned int real_yawn = 1;  //����ʱ��
	unsigned int detect_no_face_duration = 1; //δ������ʱ�� done!

	//���죺MAR>0.5
	float MAR_THRESH = 0.5;
	/**********��������ͳ�ʼ��**********/


	/**********����dlib face_landmarkԤ��ģ�ͣ�18�㣩**********/
	frontal_face_detector face_detector = get_frontal_face_detector();
	shape_predictor pre_modle_18;
	deserialize("E:\\Fatigue\\driver_fatigue_detect\\driver_fatigue_detect\\18_predictor.dat") >> pre_modle_18;
	/**********����dlib face_landmarkԤ��ģ�ͣ�18�㣩**********/

	try 
	{
		VideoCapture cap(0);
		if (!cap.isOpened()) 
		{
			cout<<"δ������ͷ"<<endl;
			return 1;
		}

		while (waitKey(30) != 27)//30ms�û�δ����ESC��
		{
			Mat src;
			cap >> src;

			clock_t flame_start = clock();

			//��srcת��ΪBGR
			cv_image<bgr_pixel> img(src);

			std::vector<dlib::rectangle> faces = face_detector(img);
			std::vector<full_object_detection> shapes;

			unsigned int faceNumber = faces.size();   //������������������

			float MAR_mouth;
			float EAR_eyes;

			/**********δ��⵽����**********/
			if (faceNumber == 0)
			{
				if (detect_no_face_duration != 100)
				{
					cout << "δ��⵽����!!\t" <<"time: "<< detect_no_face_duration++<<endl;
					
				}
				else
				{
					cout << "�ϳ�ʱ��δ��⵽�������ж�ƣ�ͣ�" << endl;
					detect_no_face_duration = 1;
				}
			}
			/**********δ��⵽����**********/

			/**********��⵽����**********/
			else
			{
				detect_no_face_duration = 0;
				for (unsigned int i = 0; i < faceNumber; ++i)
				{
					shapes.emplace_back(pre_modle_18(img, faces[i]));
				}
				
				if (!shapes.empty())
				{
					/**********18������**********/
					int faceNumber = shapes.size();
					for (int j = 0; j < faceNumber; j++)
					{
						for (int i = 0; i < 18; i++)
						{
							//��18������
							cv::circle(src, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 0.5, cv::Scalar(255, 255, 0), -1);
							cv::putText(src, to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
						}
					}

					clock_t flame_end = clock();
					/**********18������**********/

					/**********�ۿ�����*********/
					//�������꣺6��
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

					//�������꣺6��
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

					//�������:6��
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
					/**********�ۿ�����*********/

					/**********EAR*********/
					float height_left_eye = (y_4 - y_2 + y_5 - y_1) / 2;		//�����ݸ�
					unsigned int length_left_eye = x_3 - x_0;   //���ۺ��


					if (height_left_eye == 0)  //���۾��պϵ�ʱ���ݸ�Ϊ0����ʱ���¸�ֵΪ1
						height_left_eye = 1;
					float EAR_left_eye;			//���ۿ�߱�
					EAR_left_eye = height_left_eye / length_left_eye;

					float height_right_eye = (y_10 - y_8 + y_11 - y_7) / 2;		//�����ݸ�
					unsigned int length_right_eye = x_9 - x_6;     //���ۺ��
					if (height_right_eye == 0)  //���۾��պϵ�ʱ���ݸ�Ϊ0����ʱ���¸�ֵΪ1
						height_right_eye = 1;
					float EAR_right_eye;			//���ۿ�߱�
					EAR_right_eye = height_right_eye / length_right_eye;

					//EAR
					EAR_eyes = (EAR_left_eye + EAR_right_eye) / 2;
					/**********EAR*********/

					/**********MAR*********/
					unsigned int lenght_mouth = x_15 - x_12;			//��ͺ��
					float height_mouth = (y_17 - y_13 + y_16 - y_14) / 2;//����ݸ�
					
					MAR_mouth = height_mouth / lenght_mouth;//��͵Ŀ�߱�
					/**********MAR*********/



					//�������
					if (MAR_mouth > MAR_THRESH)//��ֵ0.5
					{
						open_mou_cnt++;
					}

					//���ۼ���
					if (EAR_eyes < 0.2)
					{
						close_eye_cnt++;
					}

					//գ�ۼ���
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


					char blink_cnt_text[30];   //գ�ۼ���
					char open_mou_cnt_text[30];   //�������
					char EAR_eyes_text[30];   //EAR
					char MAR_mouth_text[30];  //MAR

					_gcvt_s(EAR_eyes_text, EAR_eyes, 10);
					_gcvt_s(blink_cnt_text, blink_cnt, 10);
					_gcvt_s(MAR_mouth_text, MAR_mouth, 10);
					_gcvt_s(open_mou_cnt_text, open_mou_cnt, 10);

					cout << "EAR�� " << EAR_eyes_text << endl << endl;
					cout << "MAR�� " << MAR_mouth_text << endl << endl;
					cout << "գ�ۼ����� " << blink_cnt_text << endl << endl;
					cout << "��������� " << open_mou_cnt_text << endl << endl;

					//cv::putText(src, string("FPS: ") + to_string(1 / (flame_end - flame_start)), Point(10, 350), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 0), 1, LINE_AA);

				}

			}
			/**********��⵽����**********/

			//��Ƶ����ʾ
			cv::imshow("��ʻƣ�ͼ����...", src);


			//���ʱ�䴰��
			clock_t start = clock();
			clock_t finish = clock();
			double consumeTime = (double)(finish - start);


			/**********��Ƿ��Ϊ����**********/
			if ((open_mou_cnt / consumeTime) > 60)//����Ƶ�ʴ���60��/����Ϊһ�ι�Ƿ�����ִ���֤
			{
				    
					if (MAR_mouth < MAR_THRESH)//����ʱ��
					{
						open_mou_cnt = 0;
						real_yawn++;
					}
					else
					{
						open_mou_cnt++;
						cout << "��⵽��Ƿ��Ϊ����ǰ��Ƿ������" << real_yawn << endl;
					}
			}
			/**********��Ƿ��Ϊ����**********/
			
			//�������۾��棬�нϴ�覴ã����޸�...
			if ((close_eye_cnt / consumeTime) > 60000)//գ��Ƶ�ʴ���60��/����Ϊһ�α��ۣ����ִ���֤
			{
				if (EAR_eyes > 0.25)//����ʱ��
				{
					close_eye_cnt = 0;
					eye_close_duration++;
					EAR_eyes = 0;//������������ж�
				}
				else
				{
					close_eye_cnt++;
					cout << "��⵽������Ϊ����ǰ���۴�����" << eye_close_duration << endl;
				}
			}

		}

	}
	catch (serialization_error& e)
	{
		cout << "dlib Ԥ��ģ��δ�ɹ�����..." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}

}
