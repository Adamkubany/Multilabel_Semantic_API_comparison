<!DOCTYPE html>
<html>  
    <head>
        <title></title>
        <?php
        require 'multilabel_similarity.php';
        function averagePrecision($preds, $reals, $semantic, $th) {
            var_dump($preds, $reals, $semantic);
            $accumPrecision = 0;
            $gtLabelCount = count($reals);
            $predCount = 0;
            $tp = 0;

            foreach ($preds as $predkey => $predLabel) {   //if there are several labels per object
                $predLabelArr = explode(', ', $predLabel);
                //var_dump($predLabelArr);
                $predCount += 1;
                $same = 0;
                foreach ($predLabelArr as $predkeyArr => $predLabel_){
                    //$predLabel_ = trim($predLabel_);
                    //var_dump($predLabel_);
                    $predVec = ($semantic)? get_word_vec($predLabel_) : '';
                    foreach ($reals as $key => $label){
                        //var_dump($label);
                        if ($semantic){
                            $sim = cosine_similarity($predVec, get_word_vec($label));
                            if ($sim > $th){
                                $same = 1 ;
                            }
                            var_dump('SEM', $predLabel_, $label, $same, $sim, $predVec);

                        }
                        if (!$semantic){
                            if (strcasecmp($predLabel_, $label) == 0){
                                $same = 1;
                                }
                            var_dump('not sem', $predLabel_, $label, $same);
                        }
                    }
                }
                if ($same == 1){
                    var_dump('exist', $predLabel, $label, $same, $semantic);

                    $tp += 1;
                    $curPrecision = $tp/$predCount;
                }
                else{
                    $curPrecision = 0;
                }

                var_dump('semi count',$tp, $predCount, $accumPrecision, $curPrecision);
                $accumPrecision = $accumPrecision + $curPrecision;
                var_dump('sum', $accumPrecision);
            }
            $avgPrecision = $accumPrecision / $gtLabelCount;
            var_dump('all',$avgPrecision);
            return $avgPrecision;
        }
        function setObjVec($ObjArry, $real) {
            global $allObjects;
            global $numOfObjects;
            global $objMatrixReal;
            global $objMatrixPred;
            global $imagesNumer;
            //global $objMatrixPredNotInReal;
            //global $allObjectsPredNotInReal;

            $objVec = array_fill(0, $numOfObjects, 0);
            //for all the objects in the image
            foreach ($ObjArry as $v1) {
                // $toCheck = 1;
                $noSpaceV1 = str_replace(' ', '', $v1);
                //for all the objects
                $counterObj = 0;
                foreach ($allObjects as $k2 => $v2) {
                    //isset($objMatrixReal[$k2][$cur_img_id])? : $objMatrixReal[$k2][$cur_img_id] = 0;
                    //isset($objMatrixPred[$k2][$cur_img_id])? : $objMatrixPred[$k2][$cur_img_id] = 0;
                    $noSpaceV2 = str_replace(' ', '', $v2);
                    if ($real) {
                        //    if ($toCheck) {
                        $nameEqual = ($noSpaceV1 == $noSpaceV2) ? 1 : 0;
                        if ($nameEqual) {
                            $objVec[$k2] = 1;
                            $objMatrixReal[$k2][$imagesNumer] = 1;
                            // $toCheck = 0;
                            break;
                        }
                        //    }
                    } else {//not real - the predicted vector
                        $checkedObjNames = explode(",", $noSpaceV1);
                        $baseObjNames = explode(",", $noSpaceV2);
                        foreach ($checkedObjNames as $v3) {
                            //     if ($toCheck) {
                            $nameEqual = in_array($v3, $baseObjNames) + 0;
                            if ($nameEqual) {
                                $objMatrixPred[$k2][$imagesNumer] = 1;
                                $objVec[$k2] = 1;
                                //   $toCheck = 0;
                                break 2;
                            }
                            //      }
                        }
                        /*
                          if (++$counterObj == $numOfObjects) {
                          $counterPredObj = 0;
                          foreach ($allObjectsPredNotInReal as $k4 => $v4) {
                          $noSpaceV4 = str_replace(' ', '', $v4);
                          $nameEqual = ($noSpaceV1 == $noSpaceV4) ? 1 : 0;
                          if ($nameEqual) {
                          $objMatrixPredNotInReal[$k4][$imagesNumer] = 1;
                          break;
                          }
                          if (++$counterPredObj == count($allObjectsPredNotInReal)) {
                          $allObjectsPredNotInReal[] = $noSpaceV1;
                          $objMatrixPredNotInReal[$k4 + 1][$imagesNumer] = 1;
                          }
                          }
                          } */
                    }
                }
            }
            /*
              global $objVecOnes;
              echo 'real';
              var_dump($objMatrixReal[26][61519]);
              echo 'pred';
              var_dump($objMatrixPred[26][61519]);
              echo 'count';
              var_dump(count(array_intersect_assoc($objVec, $objVecOnes)));
              echo 'obj in image';
              var_dump($objVec[26]);
             * 
             */
            return $objVec;
        }
        ?>
    </head>
    <body>
        <?php
//echo "start time: ".microtime();
        set_time_limit(864000);
        $con = mysqli_connect("localhost", "root", "", "infomedia");
        $APIs = ['1000new_img_1_imagga', '1000new_img_2_ibmwatson', '1000new_img_4_clarifai',
            '1000new_img_5_microsoft_oxford', '1000new_img_6_wolfram', '1000new_img_7_googlevision', '1000new_img_8_caffe',
            '1000new_img_9_deepdetect', '1000new_img_10_overfeat', '1000new_img_11_tensorflow','1000new_img_12_InceptionResNetV2',
            '1000new_img_13_mobilenet_v2', '1000new_img_14_yolo_v3', '1000new_img_15_resnet_coco', '1000new_img_16_yolo_v3_coco',
            '1000new_img_17_resnet_imgnet', '1000new_img_18_vgg19'];
        $APIs = ['1000new_img_18_vgg19'];
        $topPredOptions = [5, 3, 1];
//        $topPredOptions = [1];
        $imagesTable = '1000new_images';
        $imgObjDistTable = '1000new_img_objects_dist';
        $allObjectsTable = '1000new_img_allobj';
//        $save_path = "C:/Users/Adam/Google Drive/!PhD/~Projects/03.labels similarity measurements/code/results";
        $save_path = dirname(__DIR__) . '\code\results';
        $save_path = str_replace("\\", '/', $save_path);

        //$save_path = "./results";
        $writeToFile = 1;
        ($writeToFile) ? $fpAllMeas = fopen("$save_path/all_Matrics.csv", 'w') : "";
        //($writeToFile) ? fputcsv($fpAllMeas, ['api', 'top', 'numOfImgs', 'numOfObjs', 'Hamming_Loss', 'classification_accuracy',
          //          'Accuracy', 'Precision', 'Recall', 'F1', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'Precision_Micro',
           //         'Recall_Micro', 'F1_Micro']) : "";
        ($writeToFile) ? fputcsv($fpAllMeas, ['api', 'top', 'numOfImgs', 'numOfObjs', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'Precision_Micro',
                    'Recall_Micro', 'F1_Micro']) : "";
#######for testing
//        $imagesTable = 'test_images';
//        $imgObjDistTable = 'test_img_objects_dist';
//        $allObjectsTable = 'test_img_allobjects';
        foreach ($topPredOptions as $topValue) {//for all the different top predictions
            foreach ($APIs as $APIname) {//for all the APIs
                $apiObjTable = $APIname;
                $topPredObj = $topValue;
                $imagesNumer = 0;
//getting the files ready
                //($writeToFile) ? $fpImgMeas = fopen("$save_path/ImgMeas$apiObjTable$topPredObj.csv", 'w') : "";
                //($writeToFile) ? fputcsv($fpImgMeas, ["img_id", "Hamming_Loss", "classification_accuracy", "Accuracy", "Precision", "Recall", "F1"]) : "";
                ($writeToFile) ? $fpObjMeas = fopen("$save_path/ObjMeas$apiObjTable$topPredObj.csv", 'w') : "";
                ($writeToFile) ? fputcsv($fpObjMeas, ["Label_id", "Label", "Precision_Macro", "Recall_Macro", "F1_Macro"]) : "";

                echo "API: $apiObjTable, top $topPredObj<br>";
                //getting the object list for this run
                $allObjects = [];
                $allObjSql = "SELECT distinct names FROM $imgObjDistTable where img_id in (SELECT distinct img_id FROM $apiObjTable where conf_level!='0')";
                $objTbl = mysqli_query($con, $allObjSql);
                while ($row3 = mysqli_fetch_array($objTbl)) {
                    $allObjects[] = $row3['names'];
                }
                $numOfObjects = count($allObjects);

                //how many images in the API
                $allImagesSql = "SELECT count(distinct img_id) FROM $apiObjTable where conf_level!='0'";
                $allImagesTbl = mysqli_query($con, $allImagesSql);
                $numOfImages = mysqli_fetch_array($allImagesTbl)[0];

//array_fill(0, $numOfObjects, array_fill(0, $numOfImages, 0))
                $objMatrixReal = array_fill(0, $numOfObjects, array_fill(1, $numOfImages + 1, 0));
                $objMatrixPred = array_fill(0, $numOfObjects, array_fill(1, $numOfImages + 1, 0));
                //$objMatrixPredNotInReal = [[]];
                //$allObjectsPredNotInReal = [0];

                $imgVecOnes = array_fill(1, $numOfImages + 1, 1);
                $imgVecZeros = array_fill(1, $numOfImages + 1, 0);
                $objVecOnes = array_fill(0, $numOfObjects, 1);
                $objVecZeros = array_fill(0, $numOfObjects, 0);

          //      $measures["Hamming_Loss"] = 0;
          //      $measures["classification_accuracy"] = 0;
          //      $measures["Accuracy"] = 0;
          //      $measures["Precision"] = 0;
          //      $measures["Recall"] = 0;
          //      $measures["F1"] = 0;
                $measures["Precision_Macro"] = 0;
                $measures["Recall_Macro"] = 0;
                $measures["F1_Macro"] = 0;
                $measures["Precision_Micro"] = 0;
                $measures["Recall_Micro"] = 0;
                $measures["F1_Micro"] = 0;
                //$measures["mAp"] = 0;



                //for all predicted images 
                $imagesSql = "SELECT distinct img_id FROM $apiObjTable where conf_level!='0'";
                $images = mysqli_query($con, $imagesSql);
                while ($row = mysqli_fetch_array($images)) {
                    $imagesNumer++;
                    $realImgObj = [];
                    $predictedImgObj = [];
                    $cur_img_id = $row['img_id'];

                    ##########echo "calc for img $cur_img_id<br>";
                    //preparing the real objects for image
                    $cur_img_real_objs = mysqli_query($con, "select * from $imgObjDistTable where img_id=$cur_img_id");
                    while ($row1 = mysqli_fetch_array($cur_img_real_objs)) {
                        $cur_names = $row1['names'];
                        //dealing with images without objects in the ground truth
                        if ($cur_names[0] == 'null') {
                            $cur_names[0] = "";
                        }
                        $realImgObj[] = $cur_names;
                    }
                    //var_dump($realImgObj);
                    //preparing the predicted objects for image
                    $cur_img_pred_objs = mysqli_query($con, "select * from $apiObjTable where img_id=$cur_img_id order by conf_level DESC limit 0 ,$topPredObj");
                    while ($row2 = mysqli_fetch_array($cur_img_pred_objs)) {
                        $cur_pred_names = $row2['label'];
                        //dealing with images without objects in the ground truth
                        if ($cur_pred_names[0] == 'null') {
                            $cur_pred_names[0] = "";
                        }
                        $predictedImgObj[] = $cur_pred_names;
                    }
                    //var_dump($predictedImgObj);
                    //setting one hot obj vector per image and objMatrix
                    $realImgObjVec = setObjVec($realImgObj, 1);
                    $predictedImgObjVec = setObjVec($predictedImgObj, 0);
                    //var_dump($realImgObjVec, $predictedImgObjVec);

                    //clac per image:
                    $numRealObj = count($realImgObj);
                    $numPredObj = count($predictedImgObj);

                    $tp = count(array_intersect_assoc(array_intersect_assoc($realImgObjVec, $predictedImgObjVec), $objVecOnes));
                    $fp = $numPredObj - $tp;
                    $fn = $numRealObj - $tp;
                    //var_dump($cur_img_id, $numRealObj);
                    $ImgMeasures = [$cur_img_id];
                    //$ImgMeasures["Hamming_Loss"] = count(array_diff_assoc($realImgObjVec, $predictedImgObjVec)) / $numOfObjects;
                 //   $ImgMeasures["Hamming_Loss"] = ($fp + $fn) / $numOfObjects;
                 //   $ImgMeasures["classification_accuracy"] = ($realImgObjVec == $predictedImgObjVec) ? "1" : "0";
                 //   $ImgMeasures["Accuracy"] = ($tp / ($tp + $fp + $fn));
                 //   $ImgMeasures["Precision"] = ($tp / ($tp + $fp)); //predicted
                 //   $ImgMeasures["Recall"] = ($tp / ($tp + $fn)); //real
                 //   $ImgMeasures["F1"] = ((2 * $tp) / ($tp + $fp + $tp + $fn));
                    //$ImgMeasures["AP"] = averagePrecision($predictedImgObj, $realImgObj, False, 0);
                    //$ImgMeasures["AP_sem"] = averagePrecision($predictedImgObj, $realImgObj, True, 0.3);

                   // ($writeToFile) ? fputcsv($fpImgMeas, $ImgMeasures) : "";

                //    $measures["Hamming_Loss"] += $ImgMeasures["Hamming_Loss"];
                //    $measures["classification_accuracy"] += $ImgMeasures["classification_accuracy"];
                //    $measures["Accuracy"] += $ImgMeasures["Accuracy"];
                //    $measures["Precision"] += $ImgMeasures["Precision"]; //predicted
                //    $measures["Recall"] += $ImgMeasures["Recall"]; //real
                //    $measures["F1"] += $ImgMeasures["F1"];
                    //$measures["mAp"] += $ImgMeasures["AP"];
                    //$measures["mAp_sem"] += $ImgMeasures["AP_sem"];
                }
                echo "end of images<br>";

                //calc for all images
             //   $measures["Hamming_Loss"] /= $numOfImages;
             //   $measures["classification_accuracy"] /= $numOfImages;
             //   $measures["Accuracy"] /= $numOfImages;
             //   $measures["Precision"] /= $numOfImages;
             //   $measures["Recall"] /= $numOfImages;
             //   $measures["F1"] /= $numOfImages;
                //$measures["mAp"] /= $numOfImages;
                //$measures["mAp_sem"] /= $numOfImages;



                $overAllObjTP = 0;
                //$overAllObjTN = 0;
                $overAllObjFP = 0;
                $overAllObjFN = 0;
                //for all objects
                foreach ($allObjects as $objKey => $objValue) {
                    ########echo "calc for obj $objKey: $objValue <br>";
                    $tpObj = count(array_intersect_assoc(array_intersect_assoc($objMatrixReal[$objKey], $objMatrixPred[$objKey]), $imgVecOnes));
                    $fpObj = count(array_intersect_assoc($objMatrixPred[$objKey], $imgVecOnes)) - $tpObj;
                    $fnObj = count(array_intersect_assoc($objMatrixReal[$objKey], $imgVecOnes)) - $tpObj;

                    $curPrecisionMacro = (($tpObj + $fpObj) === 0) ? 0 : ($tpObj / ($tpObj + $fpObj));
                    $curRecallMacro = (($tpObj + $fnObj) === 0) ? 0 : ($tpObj / ($tpObj + $fnObj));

                    $ObjMeasures = [$objKey, $objValue];
                    $ObjMeasures["Precision_Macro"] = $curPrecisionMacro;
                    $ObjMeasures["Recall_Macro"] = $curRecallMacro;
                    $ObjMeasures["F1_Macro"] = (($curPrecisionMacro + $curRecallMacro) === 0) ? 0 : (2 * $curPrecisionMacro * $curRecallMacro) / ($curPrecisionMacro + $curRecallMacro);
                    ($writeToFile) ? fputcsv($fpObjMeas, $ObjMeasures) : "";

                    $measures["Precision_Macro"] += $ObjMeasures["Precision_Macro"];
                    $measures["Recall_Macro"] += $ObjMeasures["Recall_Macro"];
                    $measures["F1_Macro"] += $ObjMeasures["F1_Macro"];

                    $overAllObjTP += $tpObj;
                    $overAllObjFP += $fpObj;
                    $overAllObjFN += $fnObj;
                }
                //var_dump($overAllObjTP, $overAllObjFP, $overAllObjFN);
                $measures["Precision_Macro"] /= $numOfObjects;
                $measures["Recall_Macro"] /= $numOfObjects;
                $measures["F1_Macro"] /= $numOfObjects;
                $measures["Precision_Micro"] = $overAllObjTP / ($overAllObjTP + $overAllObjFP);
                $measures["Recall_Micro"] = $overAllObjTP / ($overAllObjTP + $overAllObjFN);
                $measures["F1_Micro"] = (2 * $measures["Precision_Micro"] * $measures["Recall_Micro"]) / ($measures["Precision_Micro"] + $measures["Recall_Micro"]);
                #########echo "API: $apiObjTable, top $topPredObj<br>";
               // ($writeToFile) ? fputcsv($fpImgMeas, ['all meas']) : "";
                ($writeToFile) ? fputcsv($fpImgMeas, $measures) : "";

                ($writeToFile) ? fputcsv($fpAllMeas, array_merge([$apiObjTable, $topPredObj, $numOfImages, $numOfObjects],$measures)) : "";
                

                //($writeToFile) ? fputcsv($fpAllMeas, $measures) : "";

              //  ($writeToFile) ? fclose($fpImgMeas) : "";
                ($writeToFile) ? fclose($fpObjMeas) : "";

                var_dump($numOfImages, $numOfObjects, $measures);
            }
        }
        ($writeToFile) ? fclose($fpAllMeas) : "";
        ?>
    </body>
</html>
