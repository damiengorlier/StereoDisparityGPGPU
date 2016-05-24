#include <iostream>

#include "costVolume.h"
#include "occlusion.h"
#include "cmdLine.h"
#include "image.h"
#include "io_png.h"

#include <fstream>

static const char* OUTFILE1 = "disparity.png";
static const char* OUTFILE2 = "disparity_occlusion.png";
static const char* OUTFILE3 = "disparity_occlusion_filled.png";
static const char* OUTFILE4 = "disparity_occlusion_filled_smoothed.png";

struct ParamGen {
	int gpgpu_acc; ///< Use of GPGPU acceleration

	// Constructor with default parameters
	ParamGen()
		: gpgpu_acc(1) {}
};

static void usage(const char* name) {
	ParamGen g;
	ParamGuidedFilter p;
	ParamOcclusion q;
	std::cerr << "Stereo Disparity through Cost Aggregation with Guided Filter\n"
		<< "Usage: " << name << " [options] im1.png im2.png dmin dmax\n\n"
		<< "Options (default values in parentheses)\n"
		<< "General parameters:\n"
		<< "    -GPU acceleration: GPGPU computation ("
		<< g.gpgpu_acc << ")\n"
		<< "Cost-volume filtering parameters:\n"
		<< "    -R radius: radius of the guided filter ("
		<< p.kernel_radius << ")\n"
		<< "    -A alpha: value of alpha (" << p.alpha << ")\n"
		<< "    -E epsilon: regularization parameter (" << p.epsilon << ")\n"
		<< "    -C tau1: max for color difference ("
		<< p.color_threshold << ")\n"
		<< "    -G tau2: max for gradient difference ("
		<< p.gradient_threshold << ")\n\n"
		<< "Occlusion detection:\n"
		<< "    -o tolDiffDisp: tolerance for left-right disp. diff. ("
		<< q.tol_disp << ")\n\n"
		<< "Densification:\n"
		<< "    -O sense: fill occlusion, sense='r':right, 'l':left\n"
		<< "    -r radius: radius of the weighted median filter ("
		<< q.median_radius << ")\n"
		<< "    -c sigmac: value of sigma_color ("
		<< q.sigma_color << ")\n"
		<< "    -s sigmas: value of sigma_space ("
		<< q.sigma_space << ")\n\n"
		<< "    -a grayMin: value of gray for min disparity (255)\n"
		<< "    -b grayMax: value of gray for max disparity (0)"
		<< std::endl;
}

void test() {
	size_t width, height, width2, height2;
	float* pix1 = io_png_read_f32_rgb("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\data\\tsukuba0_square.png", &width, &height);
	float* pix2 = io_png_read_f32_rgb("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\data\\tsukuba1_square.png", &width2, &height2);
	if (!pix1 || !pix2) {
		std::cerr << "Cannot read image file " << std::endl;
		return;
	}
	if (width != width2 || height != height2) {
		std::cerr << "The images must have the same size!" << std::endl;
		return;
	}
	Image im1(pix1, width, height);
	Image im2(pix2, width, height);

	// TESTS - Commenté == OK

	//Image gradient1 = im1.gradXGPGPU();
	//Image gradient2 = im1.gradX();

	//const float* out_grad1 = &(const_cast<Image&>(gradient1))(0, 0);
	//const float* out_grad2 = &(const_cast<Image&>(gradient2))(0, 0);

	//io_png_write_f32("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\gradXGPGPU.png", out_grad1, gradient1.width(), gradient1.height(), 1);
	//io_png_write_f32("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\gradX.png", out_grad2, gradient2.width(), gradient2.height(), 1);

	//Image r = im1.r();
	//Image r_t = r.transposeGPGPU();

	//const float* out_t1 = &(const_cast<Image&>(r))(0, 0);
	//const float* out_t2 = &(const_cast<Image&>(r_t))(0, 0);

	//io_png_write_f32("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\t_normal.png", out_t1, r.width(), r.height(), 1);
	//io_png_write_f32("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\t_transpose.png", out_t2, r_t.width(), r_t.height(), 1);

	//Image scan0 = im1.r();
	//Image scan1 = scan0.integral();
	//Image scan2 = scan0.integralGPGPU();

	//const float* out_scan0 = &(const_cast<Image&>(scan0))(0, 0);
	//const float* out_scan1 = &(const_cast<Image&>(scan1))(0, 0);
	//const float* out_scan2 = &(const_cast<Image&>(scan2))(0, 0);

	//std::ofstream out_file0("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\normal.txt");
	//if (out_file0.is_open())
	//{
	//	for (int i = 0; i < scan0.width() * scan0.height(); i++){
	//		if (i % scan0.width() == 0 && i != 0) {
	//			out_file0 << std::endl << out_scan0[i] << " ";
	//		}
	//		else {
	//			out_file0 << out_scan0[i] << " ";
	//		}
	//	}
	//	out_file0.close();
	//}

	//std::ofstream out_file1("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\integral.txt");
	//if (out_file1.is_open())
	//{
	//	for (int i = 0; i < scan1.width() * scan1.height(); i++){
	//		if (i % scan1.width() == 0 && i != 0) {
	//			out_file1 << std::endl << out_scan1[i] << " ";
	//		}
	//		else {
	//			out_file1 << out_scan1[i] << " ";
	//		}
	//	}
	//	out_file1.close();
	//}

	//std::ofstream out_file2("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\integralGPGPU.txt");
	//if (out_file2.is_open())
	//{
	//	for (int i = 0; i < scan2.width() * scan2.height(); i++){
	//		if (i % scan2.width() == 0 && i != 0) {
	//			out_file2 << std::endl << out_scan2[i] << " ";
	//		}
	//		else {
	//			out_file2 << out_scan2[i] << " ";
	//		}
	//	}
	//	out_file2.close();
	//}
}

int main(int argc, char *argv[])
{
	// TEST
	test();
	exit(2);
	//
	int grayMin = 255, grayMax = 0;
	char sense = 'r'; // Camera motion direction: 'r'=to-right, 'l'=to-left
	CmdLine cmd;

	ParamGen paramGen; // General parameters
	cmd.add(make_option('GPU', paramGen.gpgpu_acc));

	ParamGuidedFilter paramGF; // Parameters for cost-volume filtering
	cmd.add(make_option('R', paramGF.kernel_radius));
	cmd.add(make_option('A', paramGF.alpha));
	cmd.add(make_option('E', paramGF.epsilon));
	cmd.add(make_option('C', paramGF.color_threshold));
	cmd.add(make_option('G', paramGF.gradient_threshold));

	ParamOcclusion paramOcc; // Parameters for filling occlusions
	cmd.add(make_option('o', paramOcc.tol_disp)); // Detect occlusion
	cmd.add(make_option('O', sense)); // Fill occlusion
	cmd.add(make_option('r', paramOcc.median_radius));
	cmd.add(make_option('c', paramOcc.sigma_color));
	cmd.add(make_option('s', paramOcc.sigma_space));

	cmd.add(make_option('a', grayMin));
	cmd.add(make_option('b', grayMax));
	try {
		cmd.process(argc, argv);
	}
	catch (std::string str) {
		std::cerr << "Error: " << str << std::endl << std::endl;
		usage(argv[0]);
		return 1;
	}
	if (argc != 5) {
		usage(argv[0]);
		return 1;
	}
	bool detectOcc = cmd.used('o') || cmd.used('O');
	bool fillOcc = cmd.used('O');

	if (sense != 'r' && sense != 'l') {
		std::cerr << "Error: invalid camera motion direction " << sense
			<< " (must be r or l)" << std::endl;
		return 1;
	}

	// Load images
	size_t width, height, width2, height2;
	float* pix1 = io_png_read_f32_rgb(argv[1], &width, &height);
	float* pix2 = io_png_read_f32_rgb(argv[2], &width2, &height2);
	if (!pix1 || !pix2) {
		std::cerr << "Cannot read image file " << argv[pix1 ? 2 : 1] << std::endl;
		return 1;
	}
	if (width != width2 || height != height2) {
		std::cerr << "The images must have the same size!" << std::endl;
		return 1;
	}
	Image im1(pix1, width, height);
	Image im2(pix2, width, height);

	// Set disparity range
	int dMin, dMax;
	if (!((std::istringstream(argv[3]) >> dMin).eof() &&
		(std::istringstream(argv[4]) >> dMax).eof())) {
		std::cerr << "Error reading dMin or dMax" << std::endl;
		return 1;
	}
	if (dMin>dMax) {
		std::cerr << "Wrong disparity range! (dMin > dMax)" << std::endl;
		return 1;
	}

	Image disp = filter_cost_volume(im1, im2, dMin, dMax, paramGF);
	if (!save_disparity(OUTFILE1, disp, dMin, dMax, grayMin, grayMax)) {
		std::cerr << "Error writing file " << OUTFILE1 << std::endl;
		return 1;
	}

	if (detectOcc) {
		std::cout << "Detect occlusions...";
		Image disp2 = filter_cost_volume(im2, im1, -dMax, -dMin, paramGF);
		detect_occlusion(disp, disp2, static_cast<float>(dMin - 1),
			paramOcc.tol_disp);
		if (!save_disparity(OUTFILE2, disp, dMin, dMax, grayMin, grayMax))  {
			std::cerr << "Error writing file " << OUTFILE2 << std::endl;
			return 1;
		}
	}

	if (fillOcc) {
		std::cout << "Post-processing: fill occlusions" << std::endl;
		Image dispDense = disp.clone();
		if (sense == 'r')
			dispDense.fillMaxX(static_cast<float>(dMin));
		else
			dispDense.fillMinX(static_cast<float>(dMin));
		if (!save_disparity(OUTFILE3, dispDense, dMin, dMax, grayMin, grayMax)) {
			std::cerr << "Error writing file " << OUTFILE3 << std::endl;
			return 1;
		}

		std::cout << "Post-processing: smooth the disparity map" << std::endl;
		fill_occlusion(dispDense, im1.medianColor(1),
			disp, dMin, dMax, paramOcc);
		if (!save_disparity(OUTFILE4, disp, dMin, dMax, grayMin, grayMax)) {
			std::cerr << "Error writing file " << OUTFILE4 << std::endl;
			return 1;
		}
	}

	free(pix1);
	free(pix2);
	return 0;
}