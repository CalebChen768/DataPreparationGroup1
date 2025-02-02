import yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from ErrorHandler import *

# === 定义 Pipeline 生成器类 ===
class PipelineBuilder:
    

    def __init__(self, yaml_path, df:pd.DataFrame):
        self.df = df
        self.config = self.load_config(yaml_path)

        self.parse_columns()
        self.transformers = {
            "MissingValueChecker": lambda **params: MissingValueChecker(**params),
            "OutOfBoundsChecker": lambda **params: OutOfBoundsChecker(**params),
            "OutlierHandler": lambda **params: OutlierHandler(**params),
            "ScaleAdjust": lambda **params: ScaleAdjust(**params),
            "TwoStepGibberishDetector": lambda: TwoStepGibberishDetector(),
            "BERTEmbeddingTransformer": lambda: BERTEmbeddingTransformer2(),
            "OneHotEncoder": lambda categories=[list(set(self.df[i].unique())-{None, np.nan}) for i in self.categorical_cols], **params: OneHotEncoder(
                categories=categories,
                **params
            ),
        }

        self.pipeline = self.create_pipeline()

    def load_config(self, yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def parse_columns(self):
        self.numerical_cols = self.config["columns"].get("numerical", {}).get("cols", [])
        self.categorical_cols = self.config["columns"].get("categorical", {}).get("cols", [])
        self.text_cols = self.config["columns"].get("text", {}).get("cols", [])


    def create_pipeline(self):
        transformers = []

        for col_type, col_config in self.config["columns"].items():
            cols = col_config["cols"]
            steps = []

            for step in col_config["pipeline"]:
                name = step["name"]
                params = step.get("params", {})

                if name in self.transformers:
                    steps.append((name.lower(), self.transformers[name](**params)))
            
            if cols in {'numerical', 'categorical'}:
                steps.append(("aligner", AlignTransformer(original_index=self.df.index)))

            elif cols == 'text':
                for i in len(steps):
                    if steps[i][0] == 'missing_text':
                        steps.insert(i, ("aligner", AlignTransformer(original_index=self.df.index)))
                        break
                steps.append(("final_aligner", AlignTransformer(original_index=self.df.index)))
                

            pipeline = Pipeline(steps)
            transformers.append((col_type, pipeline, cols))

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        return Pipeline([
            ("preprocessor", preprocessor),
            ("to_df", FunctionTransformer(lambda X: pd.DataFrame(X))),
            ("dropper", MissDropper()),
        ])

    def get_pipeline(self):

        return self.pipeline


if __name__ == "__main__":
    df = pd.read_csv("sampled_dataframe2.csv")
    pipeline_builder = PipelineBuilder("pipeline_config.yaml", df)
    pipeline = pipeline_builder.get_pipeline()
    pipeline.fit(df)
    transformed_df = pipeline.transform(df)
    print(transformed_df)
