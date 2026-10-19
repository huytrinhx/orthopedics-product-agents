"""Admin view of the master catalog's Product Family > Tray hierarchy, as
mirrored into Postgres by backend/ingestion/seed_master_catalog.py.
Admin-only, same as the rest of catalog/document management (see
auth.dependencies.require_admin). The hierarchy itself -- which families and
trays exist, their catalog names, which tray sits under which family -- comes
from the master file and is read-only here. Editable are each row's display
name (issue #15) and each family's description for the intent classifiers
(issue #30): the generated one is shown, and an admin's own replaces it.
A re-seed overwrites neither.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import require_admin
from auth.repository import UserRecord
from catalog.family_descriptions import load_generated_family_descriptions
from catalog.models import (
    DisplayNameUpdate,
    FamilyDescriptionOut,
    FamilyDescriptionUpdate,
    ProductFamilyOut,
    ProductFamilySummaryOut,
    TrayOut,
)
from catalog.repository import (
    list_families_with_trays,
    set_family_description_override,
    set_family_display_name,
    set_tray_display_name,
)

router = APIRouter()


@router.get("/product-families", response_model=list[ProductFamilyOut])
async def list_product_families(
    admin: UserRecord = Depends(require_admin),
) -> list[ProductFamilyOut]:
    families = await list_families_with_trays()
    generated_descriptions = await load_generated_family_descriptions(families) if families else {}
    return [
        ProductFamilyOut(
            id=family.id,
            name=family.name,
            display_name=family.display_name,
            trays=[
                TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
                for tray in family.trays
            ],
            generated_description=generated_descriptions.get(family.display_name, ""),
            description_override=family.description_override,
        )
        for family in families
    ]


@router.patch("/product-families/{family_id}", response_model=ProductFamilySummaryOut)
async def update_product_family_display_name(
    family_id: uuid.UUID,
    body: DisplayNameUpdate,
    admin: UserRecord = Depends(require_admin),
) -> ProductFamilySummaryOut:
    family = await set_family_display_name(family_id, body.display_name)
    if family is None:
        raise HTTPException(status_code=404, detail="product family not found")
    return ProductFamilySummaryOut(id=family.id, name=family.name, display_name=family.display_name)


@router.patch("/product-families/{family_id}/description", response_model=FamilyDescriptionOut)
async def update_product_family_description(
    family_id: uuid.UUID,
    body: FamilyDescriptionUpdate,
    admin: UserRecord = Depends(require_admin),
) -> FamilyDescriptionOut:
    family = await set_family_description_override(family_id, body.description)
    if family is None:
        raise HTTPException(status_code=404, detail="product family not found")
    return FamilyDescriptionOut(id=family.id, description_override=family.description_override)


@router.patch("/trays/{tray_id}", response_model=TrayOut)
async def update_tray_display_name(
    tray_id: uuid.UUID,
    body: DisplayNameUpdate,
    admin: UserRecord = Depends(require_admin),
) -> TrayOut:
    tray = await set_tray_display_name(tray_id, body.display_name)
    if tray is None:
        raise HTTPException(status_code=404, detail="tray not found")
    return TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
