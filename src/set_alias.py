from mlflow import MlflowClient


MODEL_NAME = "FraudGuardModel"
MODEL_VERSION = "1"
ALIAS = "champion"


def main():
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=MODEL_VERSION,
    )

    print(
        f"✅ Alias '{ALIAS}' assigned to "
        f"{MODEL_NAME} Version {MODEL_VERSION}"
    )


if __name__ == "__main__":
    main()