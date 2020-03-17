
<?php

function custom_hungarian($matrix) {
    //from https://gist.github.com/robinvanemden/9849ee9f764e1dbb40d5
    $h = count($matrix);
    $w = count($matrix[0]);
    // If the input matrix isn't square, make it square
    // and fill the empty values with the INF, here 9999999
    if ($h < $w) {
        for ($i = $h; $i < $w; ++$i) {
            $matrix[$i] = array_fill(0, $w, 999999);
        }
    } elseif ($w < $h) {
        foreach ($matrix as &$row) {
            for ($i = $w; $i < $h; ++$i) {
                $row[$i] = 999999;
            }
        }
    }
    $h = $w = max($h, $w);
    $u = array_fill(0, $h, 0);
    $v = array_fill(0, $w, 0);
    $ind = array_fill(0, $w, -1);
    foreach (range(0, $h - 1) as $i) {
        $links = array_fill(0, $w, -1);
        $mins = array_fill(0, $w, 999999);
        $visited = array_fill(0, $w, false);
        $markedI = $i;
        $markedJ = -1;
        $j = 0;
        while (true) {
            $j = -1;
            foreach (range(0, $h - 1) as $j1) {
                if (!$visited[$j1]) {
                    $cur = $matrix[$markedI][$j1] - $u[$markedI] - $v[$j1];
                    if ($cur < $mins[$j1]) {
                        $mins[$j1] = $cur;
                        $links[$j1] = $markedJ;
                    }
                    if ($j == -1 || $mins[$j1] < $mins[$j]) {
                        $j = $j1;
                    }
                }
            }
            $delta = $mins[$j];
            foreach (range(0, $w - 1) as $j1) {
                if ($visited[$j1]) {
                    $u[$ind[$j1]] += $delta;
                    $v[$j1] -= $delta;
                } else {
                    $mins[$j1] -= $delta;
                }
            }
            $u[$i] += $delta;
            $visited[$j] = true;
            $markedJ = $j;
            $markedI = $ind[$j];
            if ($markedI == -1) {
                break;
            }
        }

        while (true) {
            if ($links[$j] != -1) {
                $ind[$j] = $ind[$links[$j]];
                $j = $links[$j];
            } else {
                break;
            }
        }
        $ind[$j] = $i;
    }
    $result = array();
    foreach (range(0, $w - 1) as $j) {
        $result[$j] = $ind[$j];
    }
    return $result;
}

function cosine_similarity($x, $y) {
    $dot = 0;
    $normX = 0;
    $normY = 0;
    for ($i = 0; $i < count($x); $i++) {
        $dot += $x[$i] * $y[$i];
        $normX += pow($x[$i], 2);
        $normY += pow($y[$i], 2);
    }
    ($normX == 0) ? $normX = 1 : $normX;
    ($normY == 0) ? $normY = 1 : $normY;
    $res = $dot / (sqrt($normX) * sqrt($normY));
    $res = round($res, 2);
    return $res;
}
function euclidean_distance($x, $y) {
    $distance = 0;
    for ($i = 0; $i < count($x); $i++) {
        $distance += pow($x[$i] - $y[$i], 2);
    }
    //($distance == 0) ? $distance = 1 : $distance;
    $res = sqrt($distance);
    $res = round($res, 2);
    return $res;
}
$word_vec_columns = "`".implode("`,`", array_merge(['word_id', 'word'], range(1, 50))).'`';

function get_word_vec($word) {
    global $word_vec_columns;
    $word = str_replace(' ', '', $word);
    $con = mysqli_connect("localhost", "root", "", "infomedia");

    $sql_p = "SELECT * FROM glove6b_words_vectors_par where word='$word'";
    $sqlVec_p = mysqli_query($con, $sql_p);
    $wordVec_p = mysqli_fetch_row($sqlVec_p);
    if ($wordVec_p == NULL) {
        $sql = "SELECT * FROM glove6b_words_vectors where word='$word'";
        $sqlVec = mysqli_query($con, $sql);
        $wordVec = mysqli_fetch_row($sqlVec);
    } else {
        $wordVec = $wordVec_p;
    }
    if ($wordVec != NULL) {
        foreach ($wordVec as $key => $value) {
            //round for faster calc and easier reading
            $wordVec[$key] = round($value, 2);
        }
    } else {
        $wordVec = array_fill(0, 52, 0);
    }
    if ($wordVec_p == NULL) {
        $vals = "";
        foreach ($wordVec as $key => $value) {
            if ($key == 1) {
                $vals .= "'" . $word . "'";
            } else {
                $vals .= "'" . $wordVec[$key] . "'";
            }
            if ($key != count($wordVec)-1){
                $vals .= ",";
            }
        }
        $sql_insert = "INSERT INTO glove6b_words_vectors_par ($word_vec_columns) VALUES ($vals)";
        mysqli_query($con, $sql_insert);
    }
    //remove id and word string
    array_shift($wordVec);
    array_shift($wordVec);
    return $wordVec;
}

function get_similarity_matrix($ground_truth, $predictions) {
    $sim_matrix = [[]];
    $sim_matrix_for_ha = [[]];
    $pred_vecs = [[]];
    $GT_vecs = [[]];
    //get_word_vec takes a long time therefore seperated for faster calc
    foreach ($predictions as $pred_key => $pred_value) {
        $pred_vecs[$pred_key] = get_word_vec($pred_value);
    }
    foreach ($ground_truth as $GT_key => $GT_value) {
        $GT_vecs[$GT_key] = get_word_vec($GT_value);
    }
    foreach ($pred_vecs as $pred_key2 => $pred_vec) {
        foreach ($GT_vecs as $GT_key2 => $GT_vec) {
            //fix garge [-1,1] to [0,1] by +1
            //setting the matrix for max total cost by 2-x
//////            $sim_matrix[$pred_key2][$GT_key2] = abs(cosine_similarity($GT_vec, $pred_vec));
            $sim_matrix[$pred_key2][$GT_key2] = abs(euclidean_distance($GT_vec, $pred_vec));

            
            //$sim_matrix_for_hu[$pred_key2][$GT_key2] = 2 - ($sim_matrix[$pred_key2][$GT_key2] + 1);
        }
    }
    $return_element = $sim_matrix;
    //$return_element[1] = $sim_matrix_for_hu;
    return $return_element;
}

function compare_similarity($gt_labels, $pred_labels, $threshold) {
## pay attention for the commented code line as added abs in get_similarity_matrix() 
## so two outputs are not necessary 

    $similarity_mat = get_similarity_matrix($gt_labels, $pred_labels);
    //sim matrix: [0] with original similarity; [1] for hungarian algo
    ///$hungarian_res = custom_hungarian($similarity_mat[1]);
    $hungarian_res = custom_hungarian($similarity_mat);

    //if more/less prediction than GT ignore any out of range
    $low_level = min(count($gt_labels), count($pred_labels)) - 1;
    foreach ($hungarian_res as $p_key => $gt_key) {
        //change predicted label if similar more than threshold
        //if ($p_key <= $low_level && $gt_key <= $low_level && $similarity_mat[0][$p_key][$gt_key] >= $threshold) { 
        if ($p_key <= $low_level && $gt_key <= $low_level && $similarity_mat[$p_key][$gt_key] >= $threshold) {

            $pred_labels[$p_key] = $gt_labels[$gt_key];
        }
    }
    return $pred_labels;
}

//require 'multilabel_similarity.php';
//$gt = ["bicycle", "child", "helmet", "road", "tree"];
//$pred = ["bike", "boy", "trail", "tree", "grass", "flower"];
//$new_pred = compare_similarity($gt,$pred,0.7);
?>