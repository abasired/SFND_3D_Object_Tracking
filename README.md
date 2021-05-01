# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


## Writeup

* The various steps followed are as follows: 
  * FP.1: matchBoundingBoxes fuction was implemented to obtain the respective Bounding boxes for distance ratio computation in subsequent frames.
   	This was implemented by counting keypoint matches in with every Bounding box and picked the one with max number of matches. Later various detector and descriptor pairs are compared to analyse respective performances.
  ```
  void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
    {
      // # BB's in each frame
      int BBp_size = prevFrame.boundingBoxes.size(); 
      int BBc_size = currFrame.boundingBoxes.size();

      // array to find matching BB
      int keypt_count[BBp_size][BBc_size] = { };

      for (auto it = matches.begin(); it != matches.end() - 1; ++it)
      {
        // using queryIdx and trainIdx members from cv::DMatch
        // obtain respective matching keypoints from two consequtive frames
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);

        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);

        bool query_found = false;
        bool train_found = false;
        std::vector<int> query_id, train_id;

        // check for all the BB in previous frame containing query keypoints
        for (int i = 0; i < BBp_size; i++) {
          if (prevFrame.boundingBoxes[i].roi.contains(query_pt)){
            query_found = true;
            query_id.push_back(i);
          }
        }
        // check for all the BB in current frame containing train keypoints
        for (int i = 0; i < BBc_size; i++) {
          if (currFrame.boundingBoxes[i].roi.contains(train_pt)){
            train_found= true;
            train_id.push_back(i);
          }
        }

        // Each element in the keypt_count array corresponds to matching BB pair.
        if (query_found && train_found) 
        {
          for (auto id_prev: query_id)
            for (auto id_curr: train_id)
              keypt_count[id_prev][id_curr] += 1;
        }

      }// End of keypoint matches loop

      // Max index for every row index denotes the best matching Bounding boxes 
       for (int i = 0; i < BBp_size; i++)
      {  
        int max_count = 0;
        int id_max = 0;
        for (int j = 0; j < BBc_size; j++)
          if (keypt_count[i][j] > max_count)
          {  
            max_count = keypt_count[i][j];
            id_max = j;
          }
        bbBestMatches[i] = id_max;
      }

      // Print matching bounding boxes
      /*
      bool bMsg = true;
        if (bMsg)
            for (int i = 0; i < BBp_size; i++)
                 cout << "Box " << i << " matches " << bbBestMatches[i]<< " box" << endl;
      */
    }
  ```
  * FP.2: Using cropped lidar points and their respective projection onto 2D iamges obtained from camera, TTC is computed. These Lidar points are associated with bounding boxes detected using Yolo based Deeplearning alogorithm. Using the matching bounding boxes and the criteria that vehicle in front is in the ego lane, the appropiate lidar points are sorted based on the x-coord. Inorder to avoid outliers, used the median of the sorted x-coord to compute TTC
  
  ```
        void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                         std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
    {
      double deltaT = 1 / frameRate;
      // Lane width
      double lW = 4.0; 
      vector<double> xPrev, xCurr;
      // find Lidar points within ego lane
      for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
      {
        if (abs(it->y) <= lW / 2.0)
        { // 3D point within ego lane
          xPrev.push_back(it->x);
        }
      }

      for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
      {
        if (abs(it->y) <= lW/ 2.0)
        { // 3D point within ego lane
          xCurr.push_back(it->x);
        }
      }

      double minXPrev = 0; 
      double minXCurr = 0;

      // use median to get a good approxiamtion of the closest point
      if (xPrev.size() > 0)
      {  
        std::sort(xPrev.begin(), xPrev.end());
        long medIndex = floor(xPrev.size()/2.0);
        minXPrev = xPrev.size() % 2 == 0 ? (xPrev[medIndex - 1] + xPrev[medIndex]) / 2.0 : xPrev[medIndex];
      }

      if (xCurr.size() > 0)
      {
        std::sort(xCurr.begin(), xCurr.end());
        long medIndex = floor(xCurr.size()/2.0);
        minXCurr = xCurr.size() % 2 == 0 ? (xCurr[medIndex - 1] + xCurr[medIndex]) / 2.0 : xCurr[medIndex];
      }  

      //cout << "minXCurr: " << minXCurr << endl;
      //cout << "minXPrev: " << minXPrev << endl;
      TTC = minXCurr * deltaT / (minXPrev - minXCurr);

      //cout << " using Lidar ttc = " << TTC << "s" << endl;
      //cout << " using Lidar vel = " << minXCurr/TTC << " m/s" << endl;

    }

  ```
  * FP.3: keypoints are obtained using declared detector/descriptor type.  Later these keypoints are associated with bounding boxes based on the coordinates. Also keypoints are matched using matchDescriptors fucntion. The bounding boxes are also associated with keypoints matches as well based the distance between the pairs of matched keypoint.
  
  ```
      // associate a given bounding box with the keypoints it contains
    void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
    {
         double dist_mean = 0;
      // Keypoint matches are not associated with bounding boxes.
      // In this function we group Keypoints into respective Bounding boxes.
        std::vector<cv::DMatch>  kptMatches_bb;
        for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
        {
            cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
            if (boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y))) 
                kptMatches_bb.push_back(*it);
         }   
        // Adjust keypoint matches using the mean distance as a measure to avoid outliers. 
        double mean_distance = 0;
        for (auto it=kptMatches_bb.begin(); it!=kptMatches_bb.end(); ++it)
        {
            mean_distance += cv::norm(kptsCurr.at(it->trainIdx).pt - kptsPrev.at(it->queryIdx).pt); 
        }
        mean_distance /= kptMatches_bb.size();

        // boundingBox with appropiate kptMatches
        for (auto it = kptMatches_bb.begin(); it!=kptMatches_bb.end(); ++it)
        {
           float dist = cv::norm(kptsCurr.at(it->trainIdx).pt - kptsPrev.at(it->queryIdx).pt);
           if (dist < 1.5*mean_distance) {
               boundingBox.kptMatches.push_back(*it);
           }
        }
    }
    
   ```

  * FP.4: TTC is computed using distance ratio of matching keypoints  within a bounding box containing lidar points as well. Even in this computation median of all distanace ratio is used. 
  
  ```
    void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,                          std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visI            // compute distance ratios between all matched ke    s
      vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and pr    ame
      for (auto it1 = kptMatches.begin(); it1 != kptMatches.end()     +it1)
      { // out      loop

        // get current keypoint and its matched partner in       . frame
        cv::KeyPoint kpOuterCurr = kptsCurr.a      rainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.      queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kpt      end(); ++it2)
              er kpt.-loop

          double minDist = 100.0; /        uired distance

          // get next keypoint and its matched        n the prev. frame
          cv::KeyPoint kpInnerCurr         .at(it2->trainIdx);
          cv::KeyPoint kpInnerPrev        v.at(it2->queryIdx);

          // comp        ces and distance ratios
          double distCurr = cv::nor        urr.pt - kpInnerCurr.pt);
          double distPrev = cv::no        Prev.pt - kpInnerPrev.pt);

          if (distPrev > std::numeric_limits<dou        lon() && distCurr >= minDis           { // avoid division by zero

                  istRatio = distCurr / distPrev;              atios.push_back(distRatio);
          }
      } // eof inner loop over all matched kpts
          // eof outer loop over all matched kpts

      //     ontinue if list of distanc          t empty
       (distR         == 0)
      {
        TTC = NA         return;
      }
      // replacement for me    Ratio
      std::sort(distRatios.begin(), dist    .end());

      long medIndex = floor(distRatios.size() / 2.0);
      double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatio    ndex]; // compute median     ratio to remove outlier influe         double dT = 1 / frameRate;
      TTC = -dT / (1 - medDistRatio);
      //cout << " kpt_matches dist ratio numbe    << distRatios.size() << "  ,medDistRatio  = " <<  medD    io << endl;
      /     "     Camera ttc = " << TTC << "s" << endl;
      cout <<  TTC << "s";
    }
  
  ```
     * FP.5: using median for lidar points in TTC computations reduces the impact of outlier issue in comparrison to using the mean of x-coord of all  lidar points.
     * FP.6: Major issues with Camera is the usage of median of distance ratio for TTC computation. 
 

## TTC computation for detector/descriptor pairs

| Detector/Descriptor    |img0|img1|img2|img3|img4|img5|img6|img7|img8|img9|img10|img11|img12|img13|img14|img15|img16|img17|img18|
| :---        			 |----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|          
| SHI-TOMASI/BRISK       |15.7|12.7|13.5|13.7|12.8|11.4|12.9|13.9|12  |`57.2`|12.77|11.76|11.4 |12.08|10.41|11.45|9.51|9.54|9.83|
| SHI-TOMASI/BRIEF       |13.8|13.1|14.1|13.3|12.1|15.1|18.3|12.3|12.2|13.3|12.2 |12.22|12.32|12.71|12.97|10.72|13.22|7.75|
| SHI-TOMASI/ORB         |13.8|13.7|11.8|12.1|12.3|14.1|13.8|12.2|11.2|13.5|11.32|12.76|12.11|11.55|10.25|12.06|13.12|9.88|
| SHI-TOMASI/FREAK       |13.7|13.1|11.4|12.5|12.3|14.2|12.5|12.8|12.1|13.0|11.96|11.82|12.34|12.66|10.29|11.31|12.82|11.28|
| FAST/BRISK             |12.5|12.3|12.52|12.6|15.9|11.9|12.9|12.2|12.3|14.3|12.77|12.00|12.1|11.6|11.31|11.05|11.12|10.29|11.65|
| FAST/BRIEF             |14.1|11.6|17.1|13.5|`29.6`|13.5|13.5|14.8|14.4|13.8|12.9 |11.52|13.62|11.04|12.65|10.72|10.91|13.96|
| FAST/ORB               |13.5|12.3|11.7|14.1|17.5|12.3|15.2|12.5|12.1|15.25|11.76|11.94|13.41|10.55|12.45|12.16|12.12|14.70|
| FAST/FREAK             |12.0|12.5|13.1|13.4|12.7|11.86|12.8|12.3|12.6|12.7|11.98|12.013|11.57|11.08|10.64|10.96|10.13|11.41|
| BRISK/BRISK            |13.0|22.6|13.5|16.1|15.2|25.1|17.3|17.2|19.32|14.88|13.07|12.65|10.93|11.66|12.27|14.79|11.28|9.83|12.06|
| BRISK/BRIEF            |13.7|21.3|16.1|24.5|18.4|`42.3`|24.1|18.3|21.3|16.5|18.9|16.9|14.47|15.87|11.03|13.60|13.87|15.60|
| BRISK/ORB              |17.1|17.1|17.8|16.2|19.1|18.3|17.05|16.4|14.98|11.64|13.42|14.95|11.65|12.38|11.77|11.12|12.97|16.99|
| BRISK/FREAK            |12.4|20.1|13.2|13.8|22.3|16.32|15.4|22.0|18.05|14.25|14.78|11.75|14.61|12.15|16.18|10.20|9.31|11.18|
| ORB/BRISK              |17.48|15.93|19.22|26.68|32.54|11.58|17.36|11.1|12.28|13.30|8.49|5416.2|8.8|9.26|11.0|9.37|17.2|19.67|
| ORB/BRIEF              |16.4|16.29|30.9|24.70|24.0|18.25|20.6|31.84|91.16|10.85|10.43|16.6|10.27|8.40|9.41|s12.64|18.76|14.77|
| ORB/ORB                |123.1|9.87|25.27|35.35|-inf|13.25|-inf|-inf|-inf|-inf|8.28|-inf|-inf|-inf|-inf|-inf|18.78|
| ORB/FREAK              |12.17|-inf|11.1|12.57|743.24|12.81|-inf|11.99|19.06|-inf|8.34|33.04|7.05|54.17|8.58|10.46|11.32|12.83|
| AZAKE/BRISK            |12.28|14.8|13.7|14.25|13.94|14.74|16.20|14.37|14.10|11.51|12.41|11.25|10.36|10.41|9.9|10.07|9.39|8.88|
| AZAKE/BRIEF            |13.37|15.56|13.15|14.59|15.12|13.40|15.52|14.42|14.2|11.9|13.03|12.13|10.2|10.0|9.05|10.22|9.39|8.85|
| AZAKE/ORB              |12.8|14.73|13.13|13.86|15.4|13.91|15.7|14.23|13.79|11.49|11.86|11.6|10.7|10.14|9.51|10.36|9.05|8.99|
| AZAKE/FREAK            |12.7|13.9|14.14|13.85|15.03|13.9|15.69|13.82|13.12|12.00|11.92|10.6|11.02|9.97|9.18|9.86|9.50|8.85|


## Comments

remote: Resolving deltas: 100% (27/27), done.
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
remote: error: Trace: e0c343fe24e8c499070da188b4d343bfda7d84618b980536888cdaba75d6aa41
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File dat/yolo/yolov3.weights is 236.52 MB; this exceeds GitHub's file size limit of 100.00 MB
To https://github.com/abasired/SFND_3D_Object_Tracking.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/abasired/SFND_3D_Object_Tracking.git'