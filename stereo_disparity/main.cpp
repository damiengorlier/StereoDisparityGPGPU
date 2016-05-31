#include <iostream>

#include "costVolume.h"
#include "occlusion.h"
#include "cmdLine.h"
#include "image.h"
#include "io_png.h"
#include "deviceProperties.cuh"

#include <fstream>
#include <string>

static const char* OUTFILE1 = "disparity.png";
static const char* OUTFILE2 = "disparity_occlusion.png";
static const char* OUTFILE3 = "disparity_occlusion_filled.png";
static const char* OUTFILE4 = "disparity_occlusion_filled_smoothed.png";

struct ParamGPU {
	int gpgpu_acc;			///< Use of GPGPU acceleration
	int block_size;			///< Standard size for the GPU block

	// Constructor with default parameters
	ParamGPU()
		: gpgpu_acc(1), block_size(16) {}
};

static void usage(const char* name) {
	ParamGPU g;
	ParamGuidedFilter p;
	ParamOcclusion q;
	std::cerr << "Stereo Disparity through Cost Aggregation with Guided Filter\n"
		<< "Usage: " << name << " [options] im1.png im2.png dmin dmax\n\n"
		<< "Options (default values in parentheses)\n"
		<< "GPU parameters:\n"
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

// For the tests
void saveImage(Image image, const char *outfile, int nChannel) {
	const float* tab = &(const_cast<Image&>(image))(0, 0);
	io_png_write_f32(outfile, tab, image.width(), image.height(), nChannel);
}

// For the tests
void saveAsTxt(Image image, const char *outfile) {
	const float* tab = &(const_cast<Image&>(image))(0, 0);
	std::ofstream file(outfile);
	if (file.is_open())
	{
		for (int i = 0; i < image.width() * image.height(); i++){
			if (i % image.width() == 0 && i != 0) {
				file << std::endl << tab[i] << " ";
			}
			else {
				file << tab[i] << " ";
			}
		}
		file.close();
	}
}

void test() {
	size_t width, height, width2, height2;
	float* pix1 = io_png_read_f32_rgb("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\data\\tsukuba0.png", &width, &height);
	float* pix2 = io_png_read_f32_rgb("C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\data\\tsukuba1.png", &width2, &height2);
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

	Image r = im1.r();
	int dMin = 0;
	int dMax = 63;
	int grayMin = 255, grayMax = 0;
	ParamGuidedFilter paramGF;
	paramGF.kernel_radius = 7;

	std::string dir = "C:\\Users\\Damien\\Documents\\Visual Studio 2013\\Projects\\stereo_disparity\\stereo_disparity\\test\\";

	freopen((dir + "_stdout.txt").c_str(), "w", stdout);

	// GPU CAPABILITIES

	printDeviceProperties();

	// RGB TO GRAY

	//std::cout << "#---------------------#" << std::endl;
	//std::cout << "#     RGB TO GRAY     #" << std::endl;
	//std::cout << "#---------------------#" << std::endl;
	//saveImage(im1.rgbToGrayGPGPU(), (dir + "rgbToGray.png").c_str(), 1);

	// OPERATORS

	//std::cout << "#-------------------#" << std::endl;
	//std::cout << "#     OPERATORS     #" << std::endl;
	//std::cout << "#-------------------#" << std::endl;
	//saveAsTxt(r + r, (dir + "plus.txt").c_str());
	//saveAsTxt(r - r, (dir + "minus.txt").c_str());
	//saveAsTxt(r * r, (dir + "mul.txt").c_str());
	//saveAsTxt(r.plusGPGPU(r), (dir + "plusGPGPU.txt").c_str());
	//saveAsTxt(r.minusGPGPU(r), (dir + "minusGPGPU.txt").c_str());
	//saveAsTxt(r.multiplyGPGPU(r), (dir + "mulGPGPU.txt").c_str());

	// GRADIENT

	//std::cout << "#------------------#" << std::endl;
	//std::cout << "#     GRADIENT     #" << std::endl;
	//std::cout << "#------------------#" << std::endl;
	//saveImage(r.gradX(), (dir + "gradX.png").c_str(), 1);
	//saveImage(r.gradXGPGPU(), (dir + "gradXGPGPU.png").c_str(), 1);

	// TRANSPOSE

	//std::cout << "#-------------------#" << std::endl;
	//std::cout << "#     TRANSPOSE     #" << std::endl;
	//std::cout << "#-------------------#" << std::endl;
	//saveImage(r, (dir + "t_normal.png").c_str(), 1);
	//saveImage(r.transposeGPGPU(), (dir + "t_transpose.png").c_str(), 1);

	// INTEGRAL

	//std::cout << "#------------------#" << std::endl;
	//std::cout << "#     INTEGRAL     #" << std::endl;
	//std::cout << "#------------------#" << std::endl;
	//saveAsTxt(r, (dir + "normal.txt").c_str());
	//saveAsTxt(r.integral(), (dir + "integral.txt").c_str());
	//saveAsTxt(r.integralGPGPU(true), (dir + "integralGPGPU.txt").c_str());

	// BOXFILTER

	//std::cout << "#-------------------#" << std::endl;
	//std::cout << "#     BOXFILTER     #" << std::endl;
	//std::cout << "#-------------------#" << std::endl;
	//// Différence causée par la gestion des bords
	//Image box = r.boxFilter(4);
	//Image boxGPGPU = r.boxFilterGPGPU(4);
	//saveImage(box, (dir + "box.png").c_str(), 1);
	//saveImage(boxGPGPU, (dir + "boxGPGPU.png").c_str(), 1);
	//saveAsTxt(box, (dir + "box.txt").c_str());
	//saveAsTxt(boxGPGPU, (dir + "boxGPGPU.txt").c_str());
	//saveAsTxt(box - boxGPGPU, (dir + "box_diff.txt").c_str());

	// COVARIANCE

	//std::cout << "#--------------------#" << std::endl;
	//std::cout << "#     COVARIANCE     #" << std::endl;
	//std::cout << "#--------------------#" << std::endl;
	//// Différence due au boxFilter
	//Image g = im1.g();
	//Image meanR = r.boxFilter(4);
	//Image meanG = g.boxFilter(4);
	////Image im1xim2 = r * g;
	////Image im1xim2GPGPU = r.multiplyGPGPU(g);
	////Image meanIm1xim2 = im1xim2.boxFilter(4);
	////Image meanIm1xim2GPGPU = im1xim2GPGPU.boxFilterGPGPU(4);
	////Image mean1xmean2 = meanR* meanG;
	////Image mean1xmean2GPGPU = meanR.multiplyGPGPU(meanG);
	////Image cov = meanIm1xim2 - mean1xmean2;
	////Image covGPGPU = meanIm1xim2GPGPU - mean1xmean2GPGPU;
	//Image cov = covariance(r, meanR, g, meanG, 4);
	//Image covGPGPU = covarianceGPGPU(r, meanR, g, meanG, 4);
	////saveAsTxt(im1xim2, (dir + "im1xim2.txt").c_str());
	////saveAsTxt(im1xim2GPGPU, (dir + "im1xim2GPGPU.txt").c_str());
	////saveAsTxt(meanIm1xim2, (dir + "meanIm1xim2.txt").c_str());
	////saveAsTxt(meanIm1xim2GPGPU, (dir + "meanIm1xim2GPGPU.txt").c_str());
	////saveAsTxt(meanIm1xim2 - meanIm1xim2GPGPU, (dir + "meanIm1xim2_diff.txt").c_str());
	////saveAsTxt(mean1xmean2, (dir + "mean1xmean2.txt").c_str());
	////saveAsTxt(mean1xmean2GPGPU, (dir + "mean1xmean2GPGPU.txt").c_str());
	//saveAsTxt(cov, (dir + "cov.txt").c_str());
	//saveAsTxt(covGPGPU, (dir + "covGPGPU.txt").c_str());
	//saveAsTxt(cov - covGPGPU, (dir + "cov_diff.txt").c_str());
	//saveImage(cov, (dir + "cov.png").c_str(), 1);
	//saveImage(covGPGPU, (dir + "covGPGPU.png").c_str(), 1);

	// COST VOLUME

	//std::cout << "#---------------------#" << std::endl;
	//std::cout << "#     COST VOLUME     #" << std::endl;
	//std::cout << "#---------------------#" << std::endl;
	//std::vector<Image> costV = cost_volume(im1, im2, dMin, dMax, paramGF);
	//std::vector<Image> costVGPGPU = cost_volume_CPU_GPGPU(im1, im2, dMin, dMax, paramGF);
	//for (std::vector<int>::size_type i = 0; i != costV.size(); i++) {
	//	saveAsTxt(costV[i], (dir + "costV_CPU_" + std::to_string(i) + ".txt").c_str());
	//	saveAsTxt(costVGPGPU[i], (dir + "costV_GPU_" + std::to_string(i) + ".txt").c_str());
	//	saveAsTxt((costV[i] - costVGPGPU[i]), (dir + "costV_diff_" + std::to_string(i) + ".txt").c_str());
	//}

	// DISPARITY

	//std::cout << "#-------------------#" << std::endl;
	//std::cout << "#     DISPARITY     #" << std::endl;
	//std::cout << "#-------------------#" << std::endl;
	//Image disp = disp_cost_volume(im1, im2, dMin, dMax, paramGF);
	//Image dispGPGPU = disp_cost_volume_CPU_GPGPU(im1, im2, dMin, dMax, paramGF);
	//save_disparity((dir + "disparity.png").c_str(), disp, dMin, dMax, grayMin, grayMax);
	//save_disparity((dir + "disparityGPGPU.png").c_str(), dispGPGPU, dMin, dMax, grayMin, grayMax);
	//saveAsTxt(disp, (dir + "disparity.txt").c_str());
	//saveAsTxt(dispGPGPU, (dir + "disparityGPGPU.txt").c_str());
	//saveAsTxt(disp - dispGPGPU, (dir + "disparity_diff.txt").c_str());

	// GUIDED FILTER

	//std::cout << "#-----------------------#" << std::endl;
	//std::cout << "#     GUIDED FILTER     #" << std::endl;
	//std::cout << "#-----------------------#" << std::endl;
	//Image filter = filter_cost_volume(im1, im2, dMin, dMax, paramGF);
	//Image filterGPGPU = filter_cost_volume_CPU_GPGPU(im1, im2, dMin, dMax, paramGF);
	//save_disparity((dir + "disp_filter.png").c_str(), filter, dMin, dMax, grayMin, grayMax);
	//save_disparity((dir + "disp_filterGPGPU.png").c_str(), filterGPGPU, dMin, dMax, grayMin, grayMax);
	//saveAsTxt(filter, (dir + "disp_filter.txt").c_str());
	//saveAsTxt(filterGPGPU, (dir + "disp_filterGPGPU.txt").c_str());
	//saveAsTxt(filter - filterGPGPU, (dir + "disp_filter_diff.txt").c_str());

	// ONLY GPU

	//std::cout << "#------------------#" << std::endl;
	//std::cout << "#     ONLY GPU     #" << std::endl;
	//std::cout << "#------------------#" << std::endl;
	//Image filter = filter_cost_volume(im1, im2, dMin, dMax, paramGF);
	//Image filterOnlyGPGPU = filter_cost_volume_GPGPU(im1, im2, dMin, dMax, paramGF);
	//save_disparity((dir + "disp_filter_Only_GPGPU.png").c_str(), filterOnlyGPGPU, dMin, dMax, grayMin, grayMax);
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

	ParamGPU paramGPU; // General parameters
	cmd.add(make_option('GPU', paramGPU.gpgpu_acc));

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