from sklearn.model_selection import ParameterGrid

from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser


def initialize_param_sets(params: dict):
    """Create various traffic analysers with various tracker types
    for evaluation 
    """
    grid = ParameterGrid({'tracker_type': params['eval_tracker_types'],
           'detection_model': params['eval_detection_models'],
            # typecasting for proper insertion into psql tbl later
           'detection_frequency': [int(i) for i in params['eval_detection_frequency']],
           'detection_iou_threshold':  [float(i) for i in params['eval_detection_iou_threshold']],
           'stop_start_iou_threshold': [float(i) for i in params['eval_stop_start_iou_threshold']]})


    eval_param_sets = {}

    for param_dict in grid: 
        eval_param_sets["tracking_analyser_" + param_dict["tracker_type"] + "_" + param_dict["detection_model"].replace("-", "_")] = param_dict

    return eval_param_sets


def create_traffic_analyser(params_to_set: dict,
                            params: dict,
                            paths: dict,
                            s3_credentials: dict,
                            verbose=True
                            ): 
    """Initialize a traffic analyser with parameters passed in params_to_set
    as opposed to default parameters specified in params
    """
    params_copy = params.copy()

    # change param_copy with values in params_to_set
    for key, value in params_to_set.items():
        params_copy[key] = value

    traffic_analyser = TrackingAnalyser(params=params_copy,
                                       paths=paths,
                                       s3_credentials=s3_credentials,
                                       detection_model=params_copy['detection_model'],
                                       tracker_type=params_copy['tracker_type'],
                                       verbose=verbose)

    return traffic_analyser
