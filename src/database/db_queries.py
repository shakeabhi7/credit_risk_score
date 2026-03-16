import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseQueries:
    """Database queries for streamlit dashboard"""

    def __init__(self,db_logger):
        self.db_logger = db_logger
        self.db = db_logger.db
    
    def is_connected(self):
        """check database connection"""

        if not self.db_logger.connected:
            logger.warning("Database not connected")
            return False
        return True

    def search_by_customer_id(self,customer_id):
        """Search prediction by customer ID"""
        if not self.is_connected():
            return None
        
        if not customer_id:
            raise ValueError("Customer_id cannot be empty")
        
        try:
            collection = self.db["predictions"]

            prediction = collection.find_one({"metadata.customer_id":customer_id})

            if prediction:
                logger.info(f"Found prediction for {customer_id}")
                return prediction
            
            logger.warning(f"No prediction found for {customer_id}")
            return None
        except Exception as e:
            logger.info(f"Error searching by customer ID: {e}")
            return None
    
    def get_recent_predictions(self,limit:int=10,days:int=7):
        """Get recent predictions from last N Days"""
        if not self.is_connected():
            return []
        if limit <= 0:
            raise ValueError("limit must be positive")
        
        try:
            collection = self.db["predictions"]

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            predictions = list(
                collection.find(
                   { "metadata.timestamp":{
                        "$gte":start_date,
                        "$lte":end_date,
                    }
                    }
                ).sort("metadata.timestamp",-1)
            .limit(limit)
            )
            logger.info(f"Retrieved {len(predictions)} recent predictions")

            return predictions
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []
        
    def get_preditions_by_class(self,predicted_class:int,limit:int=20):
        """Get predictions by predicted class"""
        if not self.is_connected():
            return []
        
        if predicted_class not in [0,1]:
            raise ValueError("predicted class must be 0 and 1")
        try:
            collections = self.db["predictions"]

            predictions = list(
                collections.find(
                    {"preditions.predicted_class":predicted_class}
                )
                .sort("metadata.timestamp",-1)
                .limit(limit)
            )

            class_name = "Good Credit" if predicted_class == 0 else "Bad Credit"

            logger.info(f"Retrieved {len(predictions)} {class_name} predictions")

            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions by class: {e}")
            return []
        
    def get_statistics(self,days:int =7):
        """Get predictions statistics using single aggregation query"""

        if not self.is_connected():
            return None
        try:
            collection = self.db["predictions"]
            end_date = datetime.now(timezone.utc)
            start_date = end_date-timedelta(days=days)

            pipeline = [
                {
                    "$match": {
                        "metadata.timestamp": {
                            "$gte": start_date,
                            "$lte": end_date,
                        }
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "good_credit": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$prediction.predicted_class", 0]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "bad_credit": {
                            "$sum": {
                                "$cond": [
                                    {"$eq": ["$prediction.predicted_class", 1]},
                                    1,
                                    0,
                                ]
                            }
                        },
                        "avg_confidence": {
                            "$avg": "$prediction.confidence"
                        },
                    }
                },
            ]

            result = list(collection.aggregate(pipeline))

            if not result:
                return {
                    "total_predictions": 0,
                    "good_credit": 0,
                    "bad_credit": 0,
                    "avg_confidence": 0,
                    "good_credit_percentage": 0,
                    "bad_credit_percentage": 0
                }
            
            data = result[0]

            total = data.get("total", 0)
            good = data.get("good_credit", 0)
            bad = data.get("bad_credit", 0)

            stats = {
                "total_predictions":total,
                "good_credit":good,
                "bad_credit":bad,
                "avg_confidence": round(data["avg_confidence"],4),
                "good_credit_percentage":round((good/total*100) if total else 0,2),
                "bad_credit_percentage":round((bad/total * 100) if total else 0,2)
                }
            
            logger.info(f"Statistics retrived for last {days} days")
            return stats
    
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return None
        
    def search_similar_profiles(self,income:float,debt:float,age:int,limit:int = 5):
        """Find similar customer profiles"""

        if not self.is_connected():
            return []
        
        if income <=0 or debt <0 or age <= 0:
            raise ValueError("Invalid input values")
        
        try:
            collection = self.db['predictions']

            income_range = income * 0.2
            debt_range = debt * 0.3
            age_range = 5

            predictions = list(
                collection.find(
                    {
                        "input_data.income": {
                            "$gte": income - income_range,
                            "$lte": income + income_range,
                        },
                        "input_data.debt": {
                            "$gte": debt - debt_range,
                            "$lte": debt + debt_range,
                        },
                        "input_data.age": {
                            "$gte": age - age_range,
                            "$lte": age + age_range,
                        },
                    }
                ).limit(limit)
            )

            logger.info(f"Found { len(predictions)} similar profiles")

            return predictions
        except Exception as e:
            logger.error(f"Error searching similar profiles: {e}")
            return []
        

    def get_predictions_by_date_range(
            self,start_date:datetime,end_date:datetime,limit:int=50):
        """Get predictions within date range"""

        if not self.is_connected():
            return []
        
        if start_date >= end_date:
            raise ValueError("Start_date must be before end_date")
        
        try:
            collection = self.db['predictions']

            predictions = list(
                collection.find(
                    {
                        "metadata.timestamp": {
                            "$gte": start_date,
                            "$lte": end_date,
                        }
                    }
                )
                .sort("metadata.timestamp", -1)
                .limit(limit)
            )
            logger.info(f"Retrived {len(predictions)} predictions in date range")

            return predictions
        except Exception as e:
            logger.error(f'Error getting preditions by date range: {e}')
            return []
        
    def get_predictions_dataframe(self,limit:int=50):
        """Fetch predictions records and convert them to pandas Dataframe"""
        if not self.is_connected():
            logger.warning("Database not connected")
            return pd.DataFrame()
        try:
            collection = self.db["predictions"]

            projection = {
                "metadata.customer_id":1,
                "input_data":1,
                "engineered_features":1,
                "prediction":1,
                "metadata.timestamp":1
            }
            cursor = (
                collection.find({},projection).sort("metadata.timestamp",-1).limit(limit)
            )
            rows = []

            for pred in cursor:
                metadata = pred.get("metadata",{})
                input_data = pred.get("input_data",{})
                engineered = pred.get("engineered_features",{})
                prediction = pred.get("prediction",{})
                metadata = pred.get("metadata",{})

                ts = metadata.get("timestamp")

                timestamp = (
                ts.strftime("%Y-%m-%d %H:%M:%S")
                if ts else None
                )
                rows.append({
                "Customer_ID": metadata.get("customer_id"),
                "Age": input_data.get("age"),
                "Income": input_data.get("income"),
                "Debt": input_data.get("debt"),
                "Credit_Limit": input_data.get("credit_limit"),
                "Credit_Used": input_data.get("credit_used"),
                "Employment_Years": input_data.get("employment_years"),

                "Debt_to_Income": round(engineered.get("debt_to_income", 0), 4),
                "Credit_Utilization": round(engineered.get("credit_utilization", 0), 4),

                "Predicted_Class":
                    "Good"
                    if prediction.get("predicted_class") == 0
                    else "Bad",

                "Probability": round(prediction.get("probability", 0), 4),
                "Confidence": round(prediction.get("confidence", 0), 4),

                "Model_Used": prediction.get("model_name"),

                "Timestamp": timestamp
                })
            df = pd.DataFrame(rows)

            logger.info(f"Created DataFrame with {len(df)} records")

            return df
        except Exception as e:
            logger.error(f"Error getting predictions dataframe: {e}")
            return pd.DataFrame()

if __name__ == "__main__": 
    from src.monitoring.logger import setup_logger 
    from src.database.prediction_logger import PredictionLogger 
    logger = setup_logger('db_queries') 

    # Test queries 
    pred_logger = PredictionLogger() 
    queries = DatabaseQueries(pred_logger) 

    if pred_logger.connected:
        logger.info(" Testing database queries...") 

        # Get statistics 
        stats = queries.get_statistics(days=7) 
        if stats: 
            logger.info(f"Statistics (last 7 days):") 
            for key, value in stats.items(): 
                logger.info(f" {key}: {value}") 

        # Get recent predictions 
        recent = queries.get_recent_predictions(limit=5)
        logger.info(f"Retrieved {len(recent)} recent predictions") 
    else: 
        logger.warning("Database not connected") 
        
    pred_logger.close()

