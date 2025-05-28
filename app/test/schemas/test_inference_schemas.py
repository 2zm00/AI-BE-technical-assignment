import pytest
from pydantic import ValidationError
from app.schemas.inference import (YearMonth, StartEndDate, EducationStartEndDate, Position, Education, TalentDataInput)

# YearMonth 모델 테스트
def test_year_month_vaild_creation():
    data = {"year": 2023, "month": 5}
    ym = YearMonth(**data)
    assert ym.year == 2023
    assert ym.month == 5

def test_year_month_optional_fields():
    ym_no_month = YearMonth(year=2024)
    assert ym_no_month.year == 2024
    assert ym_no_month.month is None

    ym_no_year = YearMonth(month=12)
    assert ym_no_year.year is None
    assert ym_no_year.month == 12

def test_year_month_invalid_type():
    with pytest.raises(ValidationError):
        YearMonth(year="not_int")
    with pytest.raises(ValidationError):
        YearMonth(month="not_int")
    
#StartEndDate 모델 테스트
def test_start_end_date_valid_creation():
    data = {
        "start" : {"year" : 2022, "month" : 1},
        "end" : {"year" : 2023, "month" : 12},
    }
    sed = StartEndDate(**data)
    assert isinstance(sed.start, YearMonth)
    assert sed.start.year == 2022
    assert sed.start.month == 1
    assert isinstance(sed.end, YearMonth)
    assert sed.end.year == 2023
    assert sed.end.month == 12

def test_start_end_date_partial_data():
    sed_only_start = StartEndDate(start = {"year": 2021})
    assert sed_only_start.start.year == 2021
    assert sed_only_start.start.month is None
    assert sed_only_start.end is None

#EducationStartEndDate 모델 테스트
def test_education_start_end_date_string_input():
    edu = Education(schoolName = "Test Univ", startEndDate= "2010 - 2012")
    assert edu.schoolName == "Test Univ"
    assert edu.startEndDate == "2010 - 2012"

def test_education_start_end_date_object_input():
    data = {
        "schoolName" : "Another Univ",
        "degreeName" : "BSc",
        "startEndDate" : {
            "startDateOn" : {"year": 2013, "month": 9},
            "endDateOn" : {"year": 2017, "month": 6}
        }
    }
    edu = Education(**data)
    assert isinstance(edu.startEndDate, EducationStartEndDate)
    assert edu.startEndDate.startDateOn.year == 2013
    assert edu.startEndDate.startDateOn.month == 9
    assert edu.startEndDate.endDateOn.year == 2017
    assert edu.startEndDate.endDateOn.month == 6

def test_education_origin_start_end_date():
    data = {
        "schoolName" : "Origin Univ",
        "originStartEndDate" : {
            "startDateOn" : {"year" : 2008},
            "endDateOn" : {"year" : 2010, "month" : 5}
        }
    }
    edu = Education(**data)
    assert isinstance(edu.originStartEndDate, EducationStartEndDate)
    assert edu.originStartEndDate.startDateOn.year == 2008
    assert edu.originStartEndDate.endDateOn.year == 2010
    assert edu.originStartEndDate.endDateOn.month == 5

# Poision 모델 테스트
def test_position_minimal_data():
    pos = Position(companyName = "Test Corp", title="Tester")
    assert pos.companyName == "Test Corp"
    assert pos.title == "Tester"
    assert pos.description is None
    assert pos.startEndDate is None
    assert pos.companyLocation is None

# TalentDataInput 모델 테스트
@pytest.fixture
def sample_talent_payload_dict() -> dict:
    return {
        "headline" : "테스트 잘하는 직업",
        "summary" : "테스트를 잘해보는 팀에 리드입니다",
        "skills" : ["AI", "Python", "Leadership"],
        "positions" : [
            {
                "companyName" : "Test Corp",
                "title" : "Sample title",
                "startEndDate" : {
                    "start" : {"year" : 2023, "month" : 3}
                }
            }
        ],
        "educations" : [
            {
                "schoolName" : "Test Univ",
                "degreeName" : "학사",
                "fieldOfStudy" : "테스트공학",
                "startEndDate" : "2010 - 2012",
                "originStartEndDate" : {
                    "startDateOn" : {"year" : 2010},
                    "endDateOn" : {"year" : 2012}
                }
            }
        ],
        "linkedinUrl" : "https://test.test.test",
        "industryname" : "테스트하기",
        "recommendations" : []
    }

def test_talend_data_input_valid_payload(sample_talent_payload_dict: dict):
    talent_data = TalentDataInput(**sample_talent_payload_dict)
    assert talent_data.headline == "테스트 잘하는 직업"
    assert len(talent_data.positions) == 1
    assert talent_data.positions[0].companyName == "Test Corp"
    assert isinstance(talent_data.positions[0].startEndDate.start, YearMonth)
    assert talent_data.positions[0].startEndDate.start.year == 2023
    assert len(talent_data.educations) == 1
    assert talent_data.educations[0].schoolName == "Test Univ"
    assert talent_data.educations[0].startEndDate == "2010 - 2012"
    assert isinstance(talent_data.educations[0].originStartEndDate, EducationStartEndDate)
    assert talent_data.educations[0].originStartEndDate.startDateOn.year == 2010
    assert talent_data.educations[0].originStartEndDate.endDateOn.year == 2012
