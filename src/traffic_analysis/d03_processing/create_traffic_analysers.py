from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser


def create_traffic_analysers(params: dict,
                             paths: dict,
                             s3_credentials: dict,
                             verbose=True):
    """Create various traffic analysers with various tracker types
    for evaluation 
    """

    traffic_analysers = {}
    analyser_name = params["traffic_analyser"].lower()

    for tracker_type in params["eval_tracker_types"]:
        for detection_model in params["eval_detection_models"]:
            
            traffic_analysers[ f"{analyser_name}_{detection_model}_{tracker_type}"] = \
                eval(params["traffic_analyser"])(params=params,
                                                 paths=paths,
                                                 s3_credentials=s3_credentials,
                                                 detection_model=detection_model,
                                                 tracker_type=tracker_type,
                                                 verbose=verbose)

    return traffic_analysers
