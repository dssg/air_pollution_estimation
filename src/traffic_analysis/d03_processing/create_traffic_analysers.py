from sklearn.model_selection import ParameterGrid

from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser


def create_traffic_analysers(params: dict,
                             paths: dict,
                             s3_credentials: dict,
                             verbose=True):
    """Create various traffic analysers with various tracker types
    for evaluation 
    """
    grid = {'tracker_type': params['eval_tracker_types'],
            'detection_model': params['eval_detection_models'],
            # typecasting for proper insertion into psql tbl later
            'detection_frequency': int(params['eval_detection_frequency']),
            'detection_iou_threshold':  float(params['eval_detection_iou_threshold']),
            'stop_start_iou_threshold': float(params['eval_stop_start_iou_threshold'])}

    param_list = list(ParameterGrid(grid))

    traffic_analysers = {}

    for param_set in param_list: 
        param_copy = params.copy()

        # change param_copy with values in param_set
        for key, value in param_set.items():
            param_copy[key] = value


        traffic_analysers[ f"tracking_analyser_{tracker_type}"] = \
            [TrackingAnalyser(params=param_copy,
                              paths=paths,
                              s3_credentials=s3_credentials,
                              detection_model=params_copy['detection_model'],
                              tracker_type=params_copy['tracker_type'],
                              verbose=verbose),
             param_set # return the values we actually changed in param_copy
            ]

    return traffic_analysers
