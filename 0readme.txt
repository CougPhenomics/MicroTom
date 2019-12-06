This the project directory. Here are some notes about each item. Please read reports/PostProcessing.html for an overview of the experiment and data collected.

Download data with:

python /home/dominik/Documents/data-science-tools/LT-db-extractor.py -c ../cppcserver-local.config -o data/vis -e MicroTom -l VIS
python /home/dominik/Documents/data-science-tools/LT-db-extractor.py -c ../cppcserver-local.config -o data/psII -e MicroTom -l PSII


- capturesettings
    :dated files contain environmental conditions in the chamber throughout the experiment
    :psII_night_IndC.prg is the PSII script that was run each night.
    :CameraSettings contains an xml file for each camera with the settings used for image capture

- data
    :genotype_map.csv is the layout of the genotypes, the plantbarcode, and the roi
    :pimframes_map.csv provides metadata about each frame of the .pim file that comes from the PSII camera
    -psII
      :each image taken labeled plantbarcode-measurementlabel-timestamp-cameralabel-frame.png
    -vis
      :each image taken labeled plantbarcode-measurementlabel-timestamp-cameralabel-frame.png
    -vistest
      :random sample set of images for testing

- debug
    :contains step by step images of the  processing for a few test images. created with plantcv.plantcv.params.debug='print'

- output
  -psII
    :output_psII_level0.csv is the initial results file from the image processing
    :output_psII_level1.csv is contains some quality control based on whether the plants grew into each other or grew out of bounds. See reports/PostProcessing.html
    -fluorescence
      :contains computed PSII parameters as images e.g. YII and NPQ
    -masks
      :contains binary mask from the image processing that is used to define objects for analysis
    -pseduocolor_images
      :false-color images of each photosynthetic parameter, labeled using the base filename from data/psII appended with the time on the Induction curve and the photosynthetic parameter. FvFm is a dark-adapted state and represents t=0. The induction curve was t=40 through t=360 seconds every 20 seconds.
  -vis
    :vis.json are the initial results form image processing
    :vis.csv-single-value-traits.csv format conversion from vis.json for single value traits, e.g. shape characteristics
    :vis.csv-multi-value-traits.csv format conversion from vis.json for multivalue traits, e.g. color histograms
    :vis.csv-single-value-traits_level1.csv filtered data after QC processing. see reports/PostProcessing.html
    :vis.csv-multi-value-traits_level1.csv filtered data after QC processing. see reports/PostProcessing.html
    -colorhist_images
      :each rgb/vis image was analyzed for color distribution and plotted on histograms saved in this folder. The data for these histograms is saved in vis.json and vis.csv-multi-value-traits_level1.csv.
    -pseudocolor_images
      :each rgb/vis image was analyzed for greenness of the plant and plotted in false color saved in this folder under the plantbarcode
    -shape_images
      :each rgb/vis image was segmented for the plant object with a shape analysis. Images outlining the analyzed shape object for each roi are saved separately in their respective plantbarcode folders. The base filename is appended with _0 and _1 for roi 0 (top) and 1 (bottom).
  -vistest
    :a sample of the vis images were analyzed to test the algorithm and outputs. The outputs are the same as those detailed under output/vis
    
- reports
  :PostProcessing.Rmd is an rmarkdown file that mixes r code and textual descriptions into an integrated report
  :PostProcessing.html is the compiled result of the rmarkdown report.

-scripts
  :psII.py is the python script used to analyze the PSII images in data/psII
  :visworkflow.py an adapted and expanded version of vis.py in order to use the plantcv.parallel subpackage for image processing
  :workflowargs.py initializes the commandline arguments in order to run visworkflow.py interactively
  :makeVideos.R is an R script for making timelapse movies of the pseudocolor output
-src
  :this folder houses functions used for processing separated into different modules
  -analysis
    :empty
  -data
    :import_snapshots.py is used to load the image files in data/ and create a dataframe of metadata. Not used in the newer visworkflow.py
  -segmentation
    :createmasks.py contains functions specific to this dataset to create masks for the vis and psII images 
  -util
    :masked_stats.py contains functions for statistics constrained to the masked area
  -viz
    :add_scalebar.py creates a scalebar on the pseudocolor images
    :custom_colormaps.py enables custom colormap for pseudocolor images

:run_workflows.sh (shell script) will run visworkflow.py on unix systems with given cmdline args. See plantcv docs for all options
:run_sampleimages.sh (shell script) to randomly subsample images from data for testing