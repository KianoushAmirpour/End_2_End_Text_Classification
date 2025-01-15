from feast import Entity, FileSource, FeatureView, Field, Project
from feast.value_type import ValueType
from feast.types import Int32, Float32, String, Array
from feast.value_type import ValueType
from datetime import timedelta

project = Project(name="feature_store", description="End to end feature store.")


question_entity = Entity(name='question_id',
                         join_keys=['id'],
                         value_type=ValueType.STRING,
                         description='A key to fetch features for each question.')

meta_features_file_source = FileSource(name='meta_features_file_source',
                                       path='data/meta_features.parquet',
                                       timestamp_field='event_timestamp')

cleaned_stemmed_file_source = FileSource(name='cleaned_stemmed_file_source',
                                          path='data/cleaned_stemmed.parquet',
                                          timestamp_field='event_timestamp')

embeddings_file_source = FileSource(name='embeddings_file_source',
                                    path='data/embeddings.parquet',
                                    timestamp_field='event_timestamp')

target_file_source = FileSource(name='target_file_source',
                                    path='data/target_df.parquet',
                                    timestamp_field='event_timestamp')

train_file_source = FileSource(name='train_file_source',
                                    path='data/train.parquet',
                                    timestamp_field='event_timestamp')


meta_features = FeatureView(
    name='meta_features_extracted_from_text',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='num_words', dtype=Int32),
        Field(name='num_unique_words', dtype=Int32),
        Field(name='num_stop_words', dtype=Int32),
        Field(name='num_title_case', dtype=Int32),
        Field(name='ave_length_words', dtype=Float32),
        Field(name='num_characters', dtype=Int32),
        Field(name='num_punctuations', dtype=Int32)
    ],
    source=meta_features_file_source,
    online=True
)

cleaned_features= FeatureView(
    name='cleaned_stemmed_version_of_texts',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='cleaned', dtype=String),
        Field(name='stemmed', dtype=String)
    ],
    source=cleaned_stemmed_file_source,
    online=True
)


embeddings_features = FeatureView(
    name='embeddings_of_text',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='embeddings', dtype=Array(Float32)),
    ],
    source=embeddings_file_source,
    online=True
)

targets = FeatureView(
    name='targets',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='label', dtype=Int32),
    ],
    source=target_file_source,
    online=True
)

raw_data = FeatureView(
    name='raw_data',
    entities=[question_entity],
    ttl=timedelta(days=20),
    schema=[
        Field(name='post', dtype=String),
    ],
    source=train_file_source,
    online=True
)