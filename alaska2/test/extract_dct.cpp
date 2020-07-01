#include <stdio.h>
#include <assert.h>

#include <cv.h>    
#include <highgui.h>

extern "C"
{
#include "jpeglib.h"
#include <setjmp.h>
}

#define DEBUG 0
#define OUTPUT_IMAGES 1

/*
 * Extract the DC terms from the specified component.
 */
IplImage *
extract_dc(j_decompress_ptr cinfo, jvirt_barray_ptr *coeffs, int ci)
{
    jpeg_component_info *ci_ptr = &cinfo->comp_info[ci];
    CvSize size = cvSize(ci_ptr->width_in_blocks, ci_ptr->height_in_blocks);
    IplImage *dc = cvCreateImage(size, IPL_DEPTH_8U, 1);
    assert(dc != NULL);

    JQUANT_TBL *tbl = ci_ptr->quant_table;
    UINT16 dc_quant = tbl->quantval[0];

#if DEBUG
    printf("DCT method: %x\n", cinfo->dct_method);
    printf
    (
        "component: %d (%d x %d blocks) sampling: (%d x %d)\n", 
        ci, 
        ci_ptr->width_in_blocks, 
        ci_ptr->height_in_blocks,
        ci_ptr->h_samp_factor, 
        ci_ptr->v_samp_factor
    );

    printf("quantization table: %d\n", ci);
    for (int i = 0; i < DCTSIZE2; ++i)
    {
        printf("% 4d ", (int)(tbl->quantval[i]));
        if ((i + 1) % 8 == 0)
            printf("\n");
    }

    printf("raw DC coefficients:\n");
#endif

    JBLOCKARRAY buf =
    (cinfo->mem->access_virt_barray)
    (
        (j_common_ptr)cinfo,
        coeffs[ci],
        0,
        ci_ptr->v_samp_factor,
        FALSE
    );
    for (int sf = 0; (JDIMENSION)sf < ci_ptr->height_in_blocks; ++sf)
    {
        for (JDIMENSION b = 0; b < ci_ptr->width_in_blocks; ++b)
        {
            int intensity = 0;

            intensity = buf[sf][b][0]*dc_quant/DCTSIZE + 128;
            intensity = MAX(0,   intensity);
            intensity = MIN(255, intensity);

            cvSet2D(dc, sf, (int)b, cvScalar(intensity));

#if DEBUG
            printf("% 2d ", buf[sf][b][0]);                        
#endif
        }
#if DEBUG
        printf("\n");
#endif
    }

    return dc;

}

IplImage *upscale_chroma(IplImage *quarter, CvSize full_size)
{
    IplImage *full = cvCreateImage(full_size, IPL_DEPTH_8U, 1);
    cvResize(quarter, full, CV_INTER_NN);
    return full;
}

GLOBAL(int)
read_JPEG_file (char * filename, IplImage **dc)
{
  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;

  struct jpeg_error_mgr jerr;
  /* More stuff */
  FILE * infile;        /* source file */

  /* In this example we want to open the input file before doing anything else,
   * so that the setjmp() error recovery below can assume the file is open.
   * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
   * requires it in order to read binary files.
   */

  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return 0;
  }

  /* Step 1: allocate and initialize JPEG decompression object */

  cinfo.err = jpeg_std_error(&jerr);

  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */

  jpeg_stdio_src(&cinfo, infile);

  /* Step 3: read file parameters with jpeg_read_header() */

  (void) jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.txt for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  jvirt_barray_ptr *coeffs = jpeg_read_coefficients(&cinfo);

  IplImage *y    = extract_dc(&cinfo, coeffs, 0);
  IplImage *cb_q = extract_dc(&cinfo, coeffs, 1);
  IplImage *cr_q = extract_dc(&cinfo, coeffs, 2);

  IplImage *cb = upscale_chroma(cb_q, cvGetSize(y));
  IplImage *cr = upscale_chroma(cr_q, cvGetSize(y));

  cvReleaseImage(&cb_q);
  cvReleaseImage(&cr_q);

#if OUTPUT_IMAGES
  cvSaveImage("y.png",   y);
  cvSaveImage("cb.png", cb);
  cvSaveImage("cr.png", cr);
#endif

  *dc = cvCreateImage(cvGetSize(y), IPL_DEPTH_8U, 3);
  assert(dc != NULL);

  cvMerge(y, cr, cb, NULL, *dc);

  cvReleaseImage(&y);
  cvReleaseImage(&cb);
  cvReleaseImage(&cr);

  /* Step 7: Finish decompression */

  (void) jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  fclose(infile);

  return 1;
}

int 
main(int argc, char **argv)
{
    int ret = 0;
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s filename.jpg\n", argv[0]);
        return 1;
    }
    IplImage *dc = NULL;
    ret = read_JPEG_file(argv[1], &dc);
    assert(dc != NULL);

    IplImage *rgb = cvCreateImage(cvGetSize(dc), IPL_DEPTH_8U, 3);
    cvCvtColor(dc, rgb, CV_YCrCb2RGB);

#if OUTPUT_IMAGES
    cvSaveImage("rgb.png", rgb);
#else
    cvNamedWindow("DC", CV_WINDOW_AUTOSIZE); 
    cvShowImage("DC", rgb);
    cvWaitKey(0);
#endif

    cvReleaseImage(&dc);
    cvReleaseImage(&rgb);

    return 0;
}