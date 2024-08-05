from typing import List
from pydantic import BaseModel
# Example schema is from http://grammar.intrinsiclabs.ai/

TEST_MESSAGES = [
            # {
            #     "role": "system",
            #     "content": EXAMPLE_SCHEMA_STRING,
            # },
            # {
            #     "role": "user",
            #     "content": f"Instructions:\n{EXAMPLE_SCHEMA_STRING}\nUser:\n2013 white prius.",
            # },
            # {
            #     "role": "user",
            #     "content": f"My dream car is a red vintage ferrari! Be creative with all the fields!",
            # },
            # {
            #     "role": "user",
            #     "content": "Build an oreo milkshake. Give me 3 steps. Make sure to include 3 dictionaries in the array. Start with [{",
            # },
            {
                "role": "user",
                "content": "Jake wants a ferrari.",
            },
        ]

EXAMPLE_SCHEMA_STRING = """Use this typescript schema as reference to output valid JSON:
interface CarAndOwner {
    car: Car;
    owner: Owner;
}
 
interface Car {
    make: string;
    model: string;
    year: number;
    colors: string[];
    features: Features;
}

interface Owner {
    firstName: string;
    lastName: string;
    age: number;
}

interface Features {
    audio: AudioFeature;
    safety: SafetyFeature;
    performance: PerformanceFeature;
}

interface AudioFeature {
    brand: string;
    speakers: number;
    hasBluetooth: boolean;
}

interface SafetyFeature {
    airbags: number;
    parkingSensors: number;
    laneAssist: number;
}

interface PerformanceFeature {
    engine: string;
    horsepower: number;
    topSpeed: number;
}"""

class AudioFeature(BaseModel):
    brand: str
    speakers: int
    hasBluetooth: bool

class SafetyFeature(BaseModel):
    airbags: int
    parkingSensors: int
    laneAssist: int

class PerformanceFeature(BaseModel):
    engine: str
    horsepower: int
    topSpeed: int

class Features(BaseModel):
    audio: AudioFeature
    safety: SafetyFeature
    performance: PerformanceFeature

class Car(BaseModel):
    make: str
    model: str
    year: int
    colors: List[str]
    features: Features

class Owner(BaseModel):
    firstName: str
    lastName: str
    age: int       

class CarAndOwner(BaseModel):
    car: Car
    owner: Owner