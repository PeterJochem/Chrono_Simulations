// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Nic Olsen
// =============================================================================
// Chrono::Granular demo using SMC method. A body whose geometry is described
// by an OBJ file is time-integrated in Chrono and interacts with a Granular
// wave tank in Chrono::Granular via the co-simulation framework.
// =============================================================================

#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <vector>
#include <string>
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/core/ChGlobal.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"

// For calling Tensorflow
#include <Python.h>
#include <boost/python.hpp>

#include <stdlib.h>

// For the database
#include "mysql_connection.h"

// For parsing the JSON file
#include <jsoncpp/json/json.h>
#include <fstream>

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
using namespace chrono;
using namespace chrono::granular;

// This is for the database - it is a unique identifier foe each experiment
// We will set this later 
int experimentID = -1;

// Globals that are the GRFs computed by the neural network
double nn_Fx = 1.0;
double nn_Fy = 1.0;
double nn_Fz = 1.0;

PyObject *pName, *pModule, *pFunc;
// PyObject *pArgs, *pValue;



void ShowUsage(std::string name) {
	std::cout << "usage: " + name + " <json_file>" << std::endl;
}

void writeMeshFrames(std::ostringstream& outstream, ChBody& body, std::string obj_name, float mesh_scaling) {
	outstream << obj_name << ",";

	// Get frame position
	ChFrame<> body_frame = body.GetFrame_REF_to_abs();
	ChQuaternion<> rot = body_frame.GetRot();
	ChVector<> pos = body_frame.GetPos();

	// Get basis vectors
	ChVector<> vx = rot.GetXaxis();
	ChVector<> vy = rot.GetYaxis();
	ChVector<> vz = rot.GetZaxis();

	// Output in order
	outstream << pos.x() << ",";
	outstream << pos.y() << ",";
	outstream << pos.z() << ",";
	outstream << vx.x() << ",";
	outstream << vx.y() << ",";
	outstream << vx.z() << ",";
	outstream << vy.x() << ",";
	outstream << vy.y() << ",";
	outstream << vy.z() << ",";
	outstream << vz.x() << ",";
	outstream << vz.y() << ",";
	outstream << vz.z() << ",";
	outstream << mesh_scaling << "," << mesh_scaling << "," << mesh_scaling;
	outstream << "\n";
}



/* This formats the foot state and grf data into the form needed for SQL 
 * Input: The 6-vector describing the foot's state, and the corresponding 3-vector GRF 
 * Return std::string - the correctly formatted SQL string
 */
std::string createTrainSQL(double time, double x, double x_dt, double y, double y_dt, double z, double z_dt, double Fx, double Fy, double Fz) {

	std::string myStr = "INSERT INTO state_grf(experiment_id, time, x, x_dt, y, y_dt, z, z_dt, Fx, Fy, Fz) VALUES (";	
	myStr = myStr + std::to_string(experimentID) + ", " +  std::to_string(time) +  ", " + std::to_string(x) + ", " + std::to_string(x_dt) + ", " 
		+ std::to_string(y) + ", " + std::to_string(y_dt) + ", " + std::to_string(z) + ", " + 
		std::to_string(z_dt) + ", " + std::to_string(Fx) + ", " + std::to_string(Fy) + ", "
		+ std::to_string(Fz) + ")";

	std::cout << myStr << '\n';

	return myStr;
}


/* This connects to the database and writes the data to it
 * Inputs: The 6-vector describing the foot's state, and the corresponding 3-vector GRF  
 * Outputs: None 
 */
void writeStateToDB(double time, double x, double x_dt, double y, double y_dt, double z, double z_dt, double Fx, double Fy, double Fz) {

	try {
		sql::Driver *driver;
		sql::Connection *con;
		sql::Statement *stmt;

		/* Create a connection */
		driver = get_driver_instance();
		con = driver->connect("tcp://127.0.0.1:3306", "root", "test"); //IP Address, user name, password

		stmt = con->createStatement();

		stmt->execute("USE hoppingRobot"); //set current database as test_db

		// Format the sql command 
		std::string myCommand = createTrainSQL(time, x, x_dt, y, y_dt, z, z_dt, Fx, Fy, Fz);
		stmt->execute(myCommand);

		delete stmt;
		delete con;
		/*According to documentation,
		  You must free the sql::Statement and sql::Connection objects explicitly using delete
		  But do not explicitly free driver, the connector object. Connector/C++ takes care of freeing that. */

	}
	catch (sql::SQLException &e) {
		cout << "# ERR: " << e.what();
		cout << " (MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}

	cout << "Successfully ended" << endl;
}

/* This function sets up the database by making sure it exists 
 * and that the right tables exist
 * Input: None
 * Returns: None
 */
void setupDatbase() {

	try {
		sql::Driver *driver;
		sql::Connection *con;
		sql::Statement *stmt;

		/* Create a connection */
		driver = get_driver_instance();
		con = driver->connect("tcp://127.0.0.1:3306", "root", "test"); //IP Address, user name, password

		stmt = con->createStatement();

		try {
			std::cout << stmt->execute("CREATE DATABASE hoppingRobot") << '\n'; // create 'test_db' database
		}
		catch (sql::SQLException &e) {

			// Check if the the database already exists
			if (e.getErrorCode() == 1007) {
				std::cout << "The databse already exists" << "\n";
				// The database already exists          
			}
		}

		stmt->execute("USE hoppingRobot"); // Set current database as hoppingRobot
		try {
			stmt->execute("CREATE TABLE state_grf(experiment_id int, time double, x double, x_dt double, y double, y_dt double, z double, z_dt double, Fx double, Fy double, Fz double)");
		}
		catch (sql::SQLException &e) {

			// Check if the the database already exists
			if (e.getErrorCode() == 1050) {
				std::cout << "The table already exists" << "\n";
				// The table already exists          
			}
		}

		try {
			// Fix me - add more info to the experiment
			stmt->execute("CREATE TABLE experiment(id int, sphere_radius float, sphere_density float, box_X float, box_Y float, box_Z float, step_size float, time_end float, grav_X float, grav_Y float, grav_Z float, normalStiffS2S float, normalStiffS2W float, normalStiffS2M float, normalDampS2S float, normalDampS2W float, normalDampS2M float, tangentStiffS2S float, tangentStiffS2W float, tangentStiffS2M float, tangentDampS2S float, tangentDampS2W float, tangentDampS2M float, static_friction_coeffS2S float, static_friction_coeffS2W float, static_friction_coeffS2M float, cohesion_ratio float, adhesion_ratio_s2w float, adhesion_ratio_s2m float, psi_T float, psi_L float)");
		}
		catch (sql::SQLException &e) {

			// Check if the the database already exists
			if (e.getErrorCode() == 1050) {
				std::cout << "The table already exists" << "\n";
			}
		}


		// Query the experiment table to setup the experiment id
		try {
			sql::ResultSet *res;
			res = stmt->executeQuery("select max(id) from experiment");
			while ( res->next() ) {

				cout << "The new experiment id is = " << res->getInt(1) << "\n"; // getInt(1) returns the first column

				// Set the global value
				experimentID = res->getInt(1) + 1;
			}
		}

		catch (sql::SQLException &e) {

			// Check if the the database already exists
			// CHANGE ERROR CODE
			if (e.getErrorCode() == 1050) {
				std::cout << "Error extracting the experiment id" << "\n";
				// The table already exists
			}
		}


		delete stmt;
		delete con;
		/*According to documentation,
		  You must free the sql::Statement and sql::Connection objects explicitly using delete
		  But do not explicitly free driver, the connector object. Connector/C++ takes care of freeing that. */
	} 
	catch (sql::SQLException &e) {
		cout << "# ERR: " << e.what();
		cout << " (MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}	
}

/* Describe this function
*/
void createNewExperiment() {

	// Read the json file to 
	Json::Value root;   // starts as "null"; will contain the root value after parsing
	// std::cin >> root;
	std::ifstream config_doc("demo_GRAN_plate.json", std::ifstream::binary);
	config_doc >> root;

	std::string sphere_radius = root["sphere_radius"].asString();
	std::string sphere_density = root["sphere_density"].asString();
	std::string box_X = root["box_X"].asString();

	std::string box_Y = root["box_Y"].asString();
	std::string box_Z = root["box_Z"].asString();
	std::string step_size = root["step_size"].asString();
	std::string time_end = root["time_end"].asString();

	std::string grav_X = root["grav_X"].asString();
	std::string grav_Y = root["grav_Y"].asString();
	std::string grav_Z  = root["grav_Z"].asString();

	std::string normalStiffS2S = root["normalStiffS2S"].asString();
	std::string normalStiffS2W = root["normalStiffS2W"].asString();
	std::string normalStiffS2M  = root["normalStiffS2M"].asString();

	std::string normalDampS2S = root["normalDampS2S"].asString();
	std::string normalDampS2W = root["normalDampS2W"].asString();
	std::string normalDampS2M = root["normalDampS2M"].asString();

	std::string tangentStiffS2S = root["tangentStiffS2S"].asString();
	std::string tangentStiffS2W = root["tangentStiffS2W"].asString();
	std::string tangentStiffS2M = root["tangentStiffS2M"].asString();

	std::string tangentDampS2S = root["tangentDampS2S"].asString();
	std::string tangentDampS2W = root["tangentDampS2W"].asString();
	std::string tangentDampS2M = root["tangentDampS2M"].asString();

	std::string static_friction_coeffS2S = root["static_friction_coeffS2S"].asString();
	std::string static_friction_coeffS2W = root["static_friction_coeffS2W"].asString();
	std::string static_friction_coeffS2M = root["static_friction_coeffS2M"].asString();

	std::string cohesion_ratio = root["cohesion_ratio"].asString();
	std::string adhesion_ratio_s2w = root["adhesion_ratio_s2w"].asString();
	std::string adhesion_ratio_s2m = root["adhesion_ratio_s2m"].asString();

	std::string psi_T = root["psi_T"].asString();
	std::string psi_L = root["psi_L"].asString();

	// Now format the string to input this into the database
	try {
		sql::Driver *driver;
		sql::Connection *con;
		sql::Statement *stmt;

		/* Create a connection */
		driver = get_driver_instance();
		con = driver->connect("tcp://127.0.0.1:3306", "root", "test"); //IP Address, user name, password

		stmt = con->createStatement();

		stmt->execute("USE hoppingRobot"); // Set current database as hoppingRobot

		std::string SQLcommand = "insert into experiment values (" + std::to_string(experimentID) + ", " + sphere_radius + 
			", " + sphere_density + ", " + box_X + ", " + box_Y + ", " + box_Z + ", "
			+ step_size + ", " + time_end + ", " + grav_X  +  ", " 
			+ grav_Y + ", " + grav_Z + ", " + normalStiffS2S + ", " + normalStiffS2W + ", " 
			+ normalStiffS2M + ", " + normalDampS2S + ", " + normalDampS2W + ", " + normalDampS2M + ", " + tangentStiffS2S + 
			", " + tangentStiffS2W + ", " + tangentStiffS2M + ", " + tangentDampS2S + ", " + tangentDampS2W + ", " 
			+ tangentDampS2M + ", " + static_friction_coeffS2S + ", " + static_friction_coeffS2W + ", " + 
			static_friction_coeffS2M + ", " + cohesion_ratio + ", " + adhesion_ratio_s2w + ", " + adhesion_ratio_s2m + ", "
			+ psi_T + ", " + psi_L + " )";

		std::cout << "\n" << "\n";
		std::cout << SQLcommand;		
		std::cout << "\n";	

		stmt->execute(SQLcommand);

		delete stmt;
		delete con;
		/* According to documentation,
		   You must free the sql::Statement and sql::Connection objects explicitly using delete
		   But do not explicitly free driver, the connector object. Connector/C++ takes care of freeing that. */
	}
	catch (sql::SQLException &e) {
		cout << "# ERR: " << e.what();
		cout << " (MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}
}

/* Describe this function
 * Input: The input vector - a 6 vector 
 * Output: The GRF 
 */
void  computeGRF(double* inVector) {
	
	PyObject *pArgs, *pValue;

	//for (int j = 0; j < 6; ++j) {
        //	std::cout << inVector[j] << "\n";
        //}

        double x = inVector[0];
	double x_dt = inVector[1];
	double y = inVector[2];	
	double y_dt = inVector[3];
	double z = inVector[4];
	double z_dt = inVector[5];
	// Input vector is a 6 vector 
        int inVectorLength = 6;

	
	char file[30] = "rtn";
	char functionName[30] = "computeGRF";
	
	char* argv[5];
	int argc = 5;
	
	/*
	PyObject *pName, *pModule, *pFunc;
	PyObject *pArgs, *pValue;
	*/
	int i;
	
		
	Py_Initialize();
	/*	
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");

	pName = PyUnicode_DecodeFSDefault(file);

	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	*/
	
	
	if (pModule != NULL) {
		//  pFunc = PyObject_GetAttrString(pModule, functionName);
		/* pFunc is a new reference */
		
		if (pFunc && PyCallable_Check(pFunc)) {
			
			pArgs = PyTuple_New(6);
			
			//std::cout << "ABOUT TO SET VARIABLES IN PYTHON" << "\n";			
			for (i = 0; i < inVectorLength; ++i) {

				pValue = PyFloat_FromDouble( inVector[i] );

				if (!pValue) {
					Py_DECREF(pArgs);
					Py_DECREF(pModule);
					fprintf(stderr, "Cannot convert argument\n");
					return;
				}

				/* pValue reference stolen here: */
				PyTuple_SetItem(pArgs, i, pValue);
			}
			
			//std::cout << "ABOUT TO START CALL TO PYTHON" << "\n";
			pValue = PyObject_CallObject(pFunc, pArgs);
			Py_DECREF(pArgs);
			//std::cout << "FINISHED CALL INTO PYTHON" << "\n";

			/*
			while (pValue == NULL) {
				continue;
			}
			*/	

			if (pValue != NULL) {
				
				// printf("Result of call: %f, %f, %f\n", PyFloat_AsDouble( PyTuple_GetItem(pValue, 0) ), PyFloat_AsDouble( PyTuple_GetItem(pValue, 1) ) , PyFloat_AsDouble( PyTuple_GetItem(pValue, 2) )  );
				// std::cout << "\n";
				
				//std::cout << "SETTING GLOBALS" << "\n";
		        	// Set the GRF variables
        			nn_Fx = PyFloat_AsDouble( PyTuple_GetItem(pValue, 0) );
        			nn_Fy = PyFloat_AsDouble( PyTuple_GetItem(pValue, 1) );
        			nn_Fz = PyFloat_AsDouble( PyTuple_GetItem(pValue, 2) );

        			//std::cout << "FINISHED SETTING GLOBALS" << "\n";

				//Py_DECREF(pValue);
			}
			else {
				//Py_DECREF(pFunc);
				//Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr,"Call failed\n");
				return;
			}
		}
		else {
			if (PyErr_Occurred())
				PyErr_Print();
			fprintf(stderr, "Cannot find function \"%s\"\n", functionName);
		}
		//Py_XDECREF(pFunc);
		//Py_DECREF(pModule);
	}
	else {
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", file);
		return;
	}
	
	/*	
	if (Py_FinalizeEx() < 0) {
		return;
	}
	*/

	/*	
	std::cout << "SETTING GLOBALS" << "\n";
	// Set the GRF variables 
	nn_Fx = PyFloat_AsDouble( PyTuple_GetItem(pValue, 0) ); 
	nn_Fy = PyFloat_AsDouble( PyTuple_GetItem(pValue, 1) );
	nn_Fz = PyFloat_AsDouble( PyTuple_GetItem(pValue, 2) );
	
	std::cout << "FINISHED SETTING GLOBALS" << "\n";
	*/

	return;
}




const double time_settle = 1;
constexpr float F_CGS_TO_SI = 1e-5;
int main(int argc, char* argv[]) {
	

	char file[30] = "rtn";
        char functionName[30] = "computeGRF";

	Py_Initialize();

        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append(\".\")");

        pName = PyUnicode_DecodeFSDefault(file);

        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
	
	pFunc = PyObject_GetAttrString(pModule, functionName);	


	// Setup the database - make the right tables/make sure they already exist 
	setupDatbase();	

	std::ofstream input_pos_vel("sim_data/output_plate_positions_and_velocities.csv");
	std::ofstream out_forces("sim_data/output_plate_forces.csv");

	sim_param_holder params;
	if (argc != 2 || ParseJSON(argv[1], params) == false) {
		ShowUsage(argv[0]);
		return 1;
	}

	float iteration_step = params.step_size;

	ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density,
			make_float3(params.box_X, params.box_Y, params.box_Z));

	ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();
	double fill_bottom = -params.box_Z / 2.0; // -200/2
	double fill_top = params.box_Z / 2.0;     // 200/4

	chrono::utils::PDSampler<float> sampler(2.4f * params.sphere_radius);
	// chrono::utils::HCPSampler<float> sampler(2.05 * params.sphere_radius);

	// leave a 4cm margin at edges of sampling
	ChVector<> hdims(params.box_X / 2 , params.box_Y / 2 , 0);
	ChVector<> center(0, 0, fill_bottom + 2.0 * params.sphere_radius);
	std::vector<ChVector<float>> body_points;

	// Shift up for bottom of box
	center.z() += 3 * params.sphere_radius;
	while (center.z() < fill_top) {
		std::cout << "Create layer at " << center.z() << std::endl;
		auto points = sampler.SampleBox(center, hdims);
		body_points.insert(body_points.end(), points.begin(), points.end());
		center.z() += 2.05 * params.sphere_radius;
	}

	apiSMC_TriMesh.setElemsPositions(body_points);

	gran_sys.set_BD_Fixed(true);
	std::function<double3(float)> pos_func_wave = [&params](float t) {
		double3 pos = {0, 0, 0};

		double t0 = 0.5;
		double freq = CH_C_PI / 4;

		if (t > t0) {
			pos.x = 0.1 * params.box_X * std::sin((t - t0) * freq);
		}
		return pos;
	};

	// gran_sys.setBDWallsMotionFunction(pos_func_wave);

	gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
	gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
	gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);

	gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
	gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
	gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

	gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
	gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
	gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);

	gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
	gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
	gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);

	gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
	gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
	gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
	gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

	gran_sys.set_fixed_stepSize(params.step_size);
	gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
	gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
	gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
	gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
	gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);

	std::string mesh_filename(GetChronoDataFile("granular/demo_GRAN_plate/plate.obj"));
	std::vector<string> mesh_filenames(1, mesh_filename);

	std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, 0.f));

	float ball_radius = 20.f;
	float length = 3.81;
	float width = 2.54;
	float thickness = 0.64;
	std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(50));

	float plate_density = 2.7;//params.sphere_density / 100.f;
	float plate_mass = (float)length * width * thickness * plate_density ;
	std::vector<float> mesh_masses(1, plate_mass);

	apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

	gran_sys.setOutputMode(params.write_mode);
	gran_sys.setVerbose(params.verbose);
	filesystem::create_directory(filesystem::path(params.output_dir));

	unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();
	std::cout << nSoupFamilies << " soup families" << std::endl;
	double* meshPosRot = new double[7 * nSoupFamilies];
	float* meshVel = new float[6 * nSoupFamilies]();

	gran_sys.initialize();

	// Create rigid ball_body simulation
	/*ChSystemSMC sys_ball;
	  sys_ball.SetContactForceModel(ChSystemSMC::ContactForceModel::Hooke);
	  sys_ball.SetTimestepperType(ChTimestepper::Type::EULER_EXPLICIT);
	  sys_ball.Set_G_acc(ChVector<>(0, 0, -980));

	  double inertia = 2.0 / 5.0 * ball_mass * ball_radius * ball_radius;
	  ChVector<> ball_initial_pos(0, 0, fill_top + ball_radius + 2 * params.sphere_radius);

	  std::shared_ptr<ChBody> ball_body(sys_ball.NewBody());
	  ball_body->SetMass(ball_mass);
	  ball_body->SetInertiaXX(ChVector<>(inertia, inertia, inertia));
	  ball_body->SetPos(ball_initial_pos);
	  ball_body->SetBodyFixed(true);
	  sys_ball.AddBody(ball_body);*/
	ChSystemSMC sys_plate;
	sys_plate.SetContactForceModel(ChSystemSMC::ContactForceModel::Hooke);
	sys_plate.SetTimestepperType(ChTimestepper::Type::EULER_EXPLICIT);
	sys_plate.Set_G_acc(ChVector<>(0, 0, -980));
	//  auto rigid_plate = std::make_shared<ChBodyEasyBox>(length, width, thickness, plate_density, true, true);
	std::shared_ptr<ChBody> rigid_plate(sys_plate.NewBody());
	rigid_plate->SetMass(plate_mass);
	rigid_plate->SetPos(ChVector<>(0,0,15));
	float inertiax = 1 / 12 * plate_mass*(thickness* thickness +width*width);
	float inertiay = 1 / 12 * plate_mass * (thickness * thickness + length * length);
	float inertiaz = 1 / 12 * plate_mass * (length * length + width * width);
	rigid_plate->SetInertiaXX(ChVector<>(inertiax, inertiay, inertiaz));
	//sys_plate.AddBody(rigid_plate);
	rigid_plate->SetBodyFixed(true);
	sys_plate.AddBody(rigid_plate); 
	unsigned int out_fps = 50;
	std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

	unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

	int currframe = 0;
	unsigned int curr_step = 0;
	gran_sys.disableMeshCollision();
	clock_t start = std::clock();
	bool plate_released = false;
	double max_z = gran_sys.get_max_z();

	// Lets us downsample data    
	bool record_data = false;

	// Create a new experiment in the database
	createNewExperiment();	

	for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {

		// Important 	    
		record_data = true;	    

		if (t >= time_settle && plate_released==false ) {
			gran_sys.enableMeshCollision();

			rigid_plate->SetBodyFixed(false);
			max_z=gran_sys.get_max_z();
			rigid_plate->SetPos(ChVector<>(0, 0, max_z + params.sphere_radius + 0.96));
			rigid_plate->SetPos_dt(ChVector<>(-0.707, 0, -0.707));
			rigid_plate->SetRot(Q_from_AngAxis(CH_C_PI/6, VECT_Y));
			plate_released = true;
			std::cout << "Releasing ball" << std::endl;
		}
		else if(t >=time_settle&& plate_released==true ) {
			rigid_plate->SetPos_dt(ChVector<>(-0.707, 0, -0.707));
			rigid_plate->SetRot(Q_from_AngAxis(CH_C_PI/6, VECT_Y));

			// std::cout << "Plate intruding" << std::endl;

			record_data = true;
		}
		auto plate_pos = rigid_plate->GetPos();
		auto plate_rot = rigid_plate->GetRot();

		auto plate_vel = rigid_plate->GetPos_dt();
		// std::cout << plate_vel << "\n";

		// This is a 3-vector - access each element as array[i]
		auto plate_ang_vel = rigid_plate->GetWvel_loc();
		plate_ang_vel = rigid_plate->GetRot().GetInverse().Rotate(plate_ang_vel);
		// std::cout << plate_ang_vel[0] << "\n";

		meshPosRot[0] = plate_pos.x();
		meshPosRot[1] = plate_pos.y();
		meshPosRot[2] = plate_pos.z();
		meshPosRot[3] = plate_rot[0];
		meshPosRot[4] = plate_rot[1];
		meshPosRot[5] = plate_rot[2];
		meshPosRot[6] = plate_rot[3];

		meshVel[0] = (float)plate_vel.x();
		meshVel[1] = (float)plate_vel.y();
		meshVel[2] = (float)plate_vel.z();
		meshVel[3] = (float)plate_vel.x();
		meshVel[4] = (float)plate_vel.y();
		meshVel[5] = (float)plate_vel.z();
		//  ball_body->SetPos_dt(ChVector<>(0,0,-10)); 
		gran_sys.meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);

		gran_sys.advance_simulation(iteration_step);
		sys_plate.DoStepDynamics(iteration_step);

		float ball_force[6];
		gran_sys.collectGeneralizedForcesOnMeshSoup(ball_force);

		// Compute the GRF
		double inVector[6];
		inVector[0] = rigid_plate->GetPos()[0];
        	inVector[1] = plate_vel[0];
        	inVector[2] = rigid_plate->GetPos()[1];
        	inVector[3] = plate_vel[1];
        	inVector[4] = rigid_plate->GetPos()[2];
        	inVector[5] = plate_vel[2];
		
		// for (int j = 0; j < 6; ++j) {
		//	std::cout << inVector[j] << "\n";
		//}
		
		// std::cout << "Calling computeGRF" << "\n";
        	computeGRF(inVector);
		// std::cout << "RETURNED" << "\n";
		
		rigid_plate->Empty_forces_accumulators();
		// This applies the force ball_force[0-2] at the location plate_pos	
		// rigid_plate->Accumulate_force(ChVector<>(ball_force[0], ball_force[1], ball_force[2]), plate_pos, false);
		 rigid_plate->Accumulate_force(ChVector<>(nn_Fx, nn_Fy, nn_Fz), plate_pos, false);

		// What to do about the torque?
		rigid_plate->Accumulate_torque(ChVector<>(ball_force[3], ball_force[4], ball_force[5]), false);


		if ( record_data == true ) {

			double x = rigid_plate->GetPos()[0];
			double x_dt = plate_vel[0];
			double y = rigid_plate->GetPos()[1];
			double y_dt = plate_vel[1];
			double z = rigid_plate->GetPos()[2];
			double z_dt = plate_vel[2];

			double Fx = rigid_plate->Get_accumulated_force()[0] * F_CGS_TO_SI;
			double Fy = rigid_plate->Get_accumulated_force()[1] * F_CGS_TO_SI; 
			double Fz = rigid_plate->Get_accumulated_force()[2] * F_CGS_TO_SI;

			// writeStateToDB(t, x, x_dt, y, y_dt, z, z_dt, Fx, Fy, Fz);
			out_forces << t << "," << Fx << "," << Fy << "," << Fz << "," << '\n';

			// This originally was... What is gran_sys.get_max_z()?
			/* out_pos << t << "," << rigid_plate->GetPos()[0] << "," << rigid_plate->GetPos()[1] << ","
			   << rigid_plate->GetPos()[2] <<","<<gran_sys.get_max_z()<<"\n";
			   */

			// 1.45339, 0, -0.320547, -0.707, 0, 0, -0.946311, -0.707,	
			input_pos_vel << t << "," << plate_rot[1]  << ", " << x << "," << x_dt << "," << y << "," << y_dt << ", " << z << "," << z_dt << "," << "\n";
		}        	

		if (curr_step % out_steps == 0) {
			std::cout << "Rendering frame " << currframe << std::endl;
			/*   char filename[100];
			     sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
			     gran_sys.writeFile(std::string(filename));

			     std::string mesh_output = std::string(filename) + "_meshframes.csv";
			     std::ofstream meshfile(mesh_output);
			     std::ostringstream outstream;
			     outstream << "mesh_name,dx,dy,dz,x1,x2,x3,y1,y2,y3,z1,z2,z3,sx,sy,sz\n";
			     writeMeshFrames(outstream, *ball_body, mesh_filename, ball_radius);
			     meshfile << outstream.str();*/
		}
	}

	clock_t end = std::clock();
	double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

	std::cout << "Time: " << total_time << " seconds" << std::endl;

	delete[] meshPosRot;
	delete[] meshVel;
	out_forces.close();
	input_pos_vel.close();
	return 0;
}
