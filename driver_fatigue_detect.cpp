/*
* @File_name:  driver_fatigue_detect.cpp
* @Description: ����dlib&opencv�����ۿڱտ��͵�ͷ��⣬ʵ��ƣ���ж�
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
/*��̬����ͷ�ļ�*/
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>


using namespace std;
using namespace dlib;
using namespace cv;

int main()
{
	/**********��������ͳ�ʼ��**********/
	unsigned int blink_cnt = 0;			   //գ�ۼ���
	unsigned int open_mou_cnt = 0;         //�������
	unsigned int close_eye_cnt = 0;        //���ۼ���

	//գ�۹��̣�EAR>0.2-- - EAR<0.2-- - EAR>0.2
	float blink_EAR_before = 0.0;		//գ��ǰ
	float blink_EAR_now = 0.2;			//գ����
	float blink_EAR_after = 0.0;		//գ�ۺ�

	//�������ʱ����EAR<0.2�ĳ���ʱ��
	unsigned int eye_close_duration = 1;  //����ʱ��
	unsigned int real_yawn = 1;  //����ʱ�� done!
	unsigned int detect_no_face_duration = 1; //δ������ʱ�� done!

	//���죺MAR>0.5
	float MAR_THRESH = 0.5;

	//��ͷ���
	unsigned int nod_cnt = 0; //��ͷ����
	unsigned int nod_total = 0; //�˯��ͷ
	int head_thresh = 8; //��ͷŷ���ǣ�head����ֵ
	
	//�������ϵ
	double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002,2.3950000000000000e+002, 0.0, 0.0, 1.0 };
	//ͼ����������ϵ
	double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };

	//��������ϵ(xy)����д͹�ֵı����ͻ���ϵ��
	Mat cam_matrix = Mat(3, 3, CV_64FC1, K);
	Mat dist_coeffs = Mat(5, 1, CV_64FC1, D);

	//# ��������ϵ(UVW)����д3D�ο���--->14��

	std::vector<Point3d> object_pts;
	object_pts.push_back(Point3d(6.825897, 6.760612, 4.402142));     //#1 ��ü��
	object_pts.push_back(Point3d(1.330353, 7.122144, 6.903745));     //#2 ��ü��
	object_pts.push_back(Point3d(-1.330353, 7.122144, 6.903745));    //#3 ��ü��
	object_pts.push_back(Point3d(-6.825897, 6.760612, 4.402142));    //#4 ��ü��
	object_pts.push_back(Point3d(5.311432, 5.485328, 3.987654));     //#7 ������
	object_pts.push_back(Point3d(1.789930, 5.393625, 4.413414));     //#10 ������
	object_pts.push_back(Point3d(-1.789930, 5.393625, 4.413414));    //#13 ������
	object_pts.push_back(Point3d(-5.311432, 5.485328, 3.987654));    //#16 ������
	object_pts.push_back(Point3d(2.005628, 1.409845, 6.165652));     //#5 ����
	object_pts.push_back(Point3d(-2.005628, 1.409845, 6.165652));    //#6 ����
	object_pts.push_back(Point3d(2.774015, -2.080775, 5.048531));    //#19 ����
	object_pts.push_back(Point3d(-2.774015, -2.080775, 5.048531));   //#22 ����
	object_pts.push_back(Point3d(0.000000, -3.116408, 6.097667));    //#24 �����м�
	object_pts.push_back(Point3d(0.000000, -7.415691, 4.070434));    //#0 �°�

	//��ά�ο��㣨ͼ�����꣩���ο���⵽���沿����
	std::vector<Point2d> image_pts;

	//result
	Mat rotation_vec;                           //3 x 1
	Mat rotation_mat;                           //3 x 3 R
	Mat translation_vec;                        //3 x 1 T
	Mat pose_mat = Mat(3, 4, CV_64FC1);     //3 x 4 R | T
	Mat euler_angle = Mat(3, 1, CV_64FC1); //ŷ���Ǿ���

	//����ͶӰ3D�����������������֤�������
	std::vector<cv::Point3d> reprojectsrc;
	reprojectsrc.push_back(Point3d(10.0, 10.0, 10.0));
	reprojectsrc.push_back(Point3d(10.0, 10.0, -10.0));
	reprojectsrc.push_back(Point3d(10.0, -10.0, -10.0));
	reprojectsrc.push_back(Point3d(10.0, -10.0, 10.0));
	reprojectsrc.push_back(Point3d(-10.0, 10.0, 10.0));
	reprojectsrc.push_back(Point3d(-10.0, 10.0, -10.0));
	reprojectsrc.push_back(Point3d(-10.0, -10.0, -10.0));
	reprojectsrc.push_back(Point3d(-10.0, -10.0, 10.0));

	//����ͶӰ�� 2D ��
	std::vector<cv::Point2d> reprojectdst;
	reprojectdst.resize(8);

	//���ڷֽ�ProjectionMatrix()ͶӰ���󣨣�����ʱ������
	Mat out_intrinsics = Mat(3, 3, CV_64FC1);
	Mat out_rotation = Mat(3, 3, CV_64FC1);
	Mat out_translation = Mat(3, 1, CV_64FC1);


	/**********��������ͳ�ʼ��**********/


	/**********����dlib face_landmarkԤ��ģ�ͣ�26�㣩**********/
	frontal_face_detector face_detector = get_frontal_face_detector();
	shape_predictor pre_model_26;
	deserialize("E:\\Fatigue\\driver_fatigue_detect\\driver_fatigue_detect\\26_predictor.dat") >> pre_model_26;
	/**********����dlib face_landmarkԤ��ģ�ͣ�26�㣩**********/

	try 
	{
		VideoCapture cap(0);
		if (!cap.isOpened()) 
		{
			cout<<"δ������ͷ"<<endl;
			return 1;
		}

		double fps;
		double t = 0;

		while (waitKey(30) != 27)//30ms�û�δ����ESC��,��֡���й�
		{
			Mat src;
			cap >> src;

			t = (double)getTickCount();

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
					shapes.emplace_back(pre_model_26(img, faces[i]));
				}
				
				if (!shapes.empty())
				{
					/**********26������**********/
					int faceNumber = shapes.size();
					for (int j = 0; j < faceNumber; j++)
					{
						for (int i = 0; i < 26; i++)
						{
							//cv::circle(src, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 0.5, cv::Scalar(255, 0, 0), -1);
							cv::putText(src, to_string(i), cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 255, 0));
						}
					}

					//��д��ά�ο���--->14��,�д��Ż�
					image_pts.push_back(Point2d(shapes[0].part(1).x(), shapes[0].part(1).y())); //#1 ��ü��
					image_pts.push_back(Point2d(shapes[0].part(2).x(), shapes[0].part(2).y())); //#2 ��ü��
					image_pts.push_back(Point2d(shapes[0].part(3).x(), shapes[0].part(3).y())); //#3 ��ü��
					image_pts.push_back(Point2d(shapes[0].part(4).x(), shapes[0].part(4).y())); //#4 ��ü��

					image_pts.push_back(Point2d(shapes[0].part(7).x(), shapes[0].part(7).y())); //#7 ������
					image_pts.push_back(Point2d(shapes[0].part(10).x(), shapes[0].part(10).y())); //#10 ������
					image_pts.push_back(Point2d(shapes[0].part(13).x(), shapes[0].part(13).y())); //#13 ������
					image_pts.push_back(Point2d(shapes[0].part(16).x(), shapes[0].part(16).y())); //#16 ������

					image_pts.push_back(Point2d(shapes[0].part(5).x(), shapes[0].part(5).y())); //#5 ����
					image_pts.push_back(Point2d(shapes[0].part(6).x(), shapes[0].part(6).y())); //#6 ����

					image_pts.push_back(Point2d(shapes[0].part(19).x(), shapes[0].part(19).y())); //#19 ����
					image_pts.push_back(Point2d(shapes[0].part(22).x(), shapes[0].part(22).y())); //#22 ����
					image_pts.push_back(Point2d(shapes[0].part(24).x(), shapes[0].part(24).y())); //#24 �����м�
					image_pts.push_back(Point2d(shapes[0].part(0).x(), shapes[0].part(0).y()));   //#0 �°�

					/**********26������**********/

					/**********�ۿ�����*********/
					//�������꣺6��
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

					//�������꣺6��
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

					//�������:6��
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
					/**********�ۿ�����*********/

					/**********EAR*********/
					float height_left_eye = (y_11 - y_9 + y_12 - y_8) / 2;		//�����ݸ�
					unsigned int length_left_eye = x_10 - x_7;   //���ۺ��


					if (height_left_eye == 0)  //���۾��պϵ�ʱ���ݸ�Ϊ0����ʱ���¸�ֵΪ1
						height_left_eye = 1;
					float EAR_left_eye;			//���ۿ�߱�
					EAR_left_eye = height_left_eye / length_left_eye;

					float height_right_eye = (y_17 - y_15 + y_18 - y_14) / 2;		//�����ݸ�
					unsigned int length_right_eye = x_16 - x_13;     //���ۺ��
					if (height_right_eye == 0)  //���۾��պϵ�ʱ���ݸ�Ϊ0����ʱ���¸�ֵΪ1
						height_right_eye = 1;
					float EAR_right_eye;			//���ۿ�߱�
					EAR_right_eye = height_right_eye / length_right_eye;

					//EAR
					EAR_eyes = (EAR_left_eye + EAR_right_eye) / 2;
					/**********EAR*********/

					/**********MAR*********/
					unsigned int lenght_mouth = x_22 - x_19;			//��ͺ��
					float height_mouth = (y_25 - y_20 + y_23 - y_21) / 2;//����ݸ�
					
					MAR_mouth = height_mouth / lenght_mouth;//��͵Ŀ�߱�
					/**********MAR*********/

					//���Ƽ��㣺3D�ο���+2D������
					solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

					//���¹滮 
					projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

					//calc euler angle �Ƕȼ���
					Rodrigues(rotation_vec, rotation_mat);
					hconcat(rotation_mat, translation_vec, pose_mat);
					decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, noArray(), noArray(), noArray(), euler_angle);

					float HEAD_THRESH = 0.3;
					double head = euler_angle.at<double>(0);

					/**********��ͷ���(����)**********/
					if (head > head_thresh)
					{
						nod_cnt += 1;
					}
					else //�������3�ζ�С����ֵ�����ʾ�˯��ͷһ��
					{
						if (nod_cnt >= 3)
						{
							nod_total += 1;
							nod_cnt = 0;
							if (nod_total > 10)
							{
								cout << "��⵽��ε�ͷ��Ϊ������ƣ�ͣ�����Ϣ��" << endl;
								nod_total = 0;
							}
						}
					}

					char nod_total_text[30];
					_gcvt_s(nod_total_text, nod_total, 10);
					cout << "��ͷ������" << nod_total_text << endl<<endl;

					//char head_text[30];
					//_gcvt_s(head_text, head, 10);
					//cout << "heading��:" << head_text << endl<<endl;

					image_pts.clear();

					/**********��ͷ���(����)**********/


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

					//cout << "EAR�� " << EAR_eyes_text << endl << endl;
					//cout << "MAR�� " << MAR_mouth_text << endl << endl;
					cout << "��ǰ���۹���գ�ۼ����� " << blink_cnt_text << endl << endl;
					cout << "��ǰ��Ƿ������������� " << open_mou_cnt_text << endl << endl;

					//cv::putText(src, string("FPS: ") + to_string(1 / (flame_end - flame_start)), Point(10, 350), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 0), 1, LINE_AA);

				}

			}
			/**********��⵽����**********/

			/*��Ƶ֡����ʾ*/
			t = ((double)getTickCount() - t) / getTickFrequency();
			fps = 1.0 / t;
			putText(src, "FPS: "+to_string(fps), cvPoint(5, 20), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 255, 0));
			/*��Ƶ֡����ʾ*/

			//��Ƶ����ʾ
			cv::imshow("��ʻƣ�ͼ����...", src);


			//���ʱ�䴰��
			clock_t start = clock();
			clock_t finish = clock();
			double consumeTime = (double)(finish - start);


			///**********��Ƿ��Ϊ����**********/
			if ((open_mou_cnt / consumeTime) > 60)//����Ƶ�ʴ���60��/����Ϊһ�ι�Ƿ�����ִ���֤
			{
				    
					if (MAR_mouth < MAR_THRESH)//����ʱ��
					{
						open_mou_cnt = 0;
						real_yawn++;
						if (real_yawn > 8)
						{
							cout << "��⵽��ι�Ƿ��Ϊ������ƣ�ͣ�����Ϣ��" << endl;
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
						cout << "��⵽��Ƿ��Ϊ����ǰ��Ƿ������" << real_yawn << endl;

					}
			}
			/**********��Ƿ��Ϊ����**********/

			/**********������Ϊ����**********/
			if ((blink_cnt/ consumeTime) >60)//գ��Ƶ�ʴ��ڴ�60/����Ϊһ�α��۹��̣����ִ���֤
			{
				if (EAR_eyes > 0.18)//����ʱ��
				{
					blink_cnt = 0;
					eye_close_duration++;
					if (eye_close_duration > 8)
					{
						cout << "��⵽��α�����Ϊ������ƣ�ͣ�����Ϣ��" << endl;
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
						cout << "��⵽������Ϊ����ǰ���۴�����" << eye_close_duration << endl;
					}
				}
			}
			///**********������Ϊ����**********/

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
